"""Grounded Answer Generation with Page-Level Citations.

Accepts fused, ranked evidence chunks (from EvidenceFuser) and a user query,
builds a strict grounding prompt, calls the configured LLM (Gemini 1.5 Flash),
and returns a structured AnswerResult with verifiable citations.

Key design properties:
- The LLM is instructed to answer ONLY from provided evidence.
- Every factual claim must be followed by an inline citation tag.
- If the evidence is insufficient, the system must say so explicitly.
- A mock generator is available for environments without a Gemini API key.
"""

import os
import re
import textwrap
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.config import AppConfig
from src.indexing.schema import ModalityType
from src.fusion.evidence_fuser import FusedEvidenceResult
from src.query_understanding.query_analyzer import QueryAnalysisResult


# ──────────────────────────────────────────────────────────────────────────────
# Answer result dataclass
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Citation:
    """Represents a single grounded page-level citation."""
    chunk_id: str
    citation_label: str        # e.g. "[Page 4, Table 2]"
    modality: str              # TEXT / TABLE / FIGURE
    page_number: int
    content_preview: str       # First 200 chars of evidence content


@dataclass
class AnswerResult:
    """Complete grounded QA result including the answer text and all citations."""
    query: str
    answer: str                          # LLM-generated grounded answer
    citations: List[Citation]            # Deduplicated evidence references
    intent: str                          # Detected intent from QueryAnalyzer
    modality_weights: Dict[str, float]   # Serialized modality weight vector
    top_evidence_count: int              # Number of evidence chunks used
    model_used: str                      # LLM model identifier
    is_fallback: bool = False            # True if mock / no-API fallback
    warning: Optional[str] = None       # Any quality warnings

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "citations": [
                {
                    "citation_label": c.citation_label,
                    "modality": c.modality,
                    "page_number": c.page_number,
                    "content_preview": c.content_preview,
                }
                for c in self.citations
            ],
            "intent": self.intent,
            "modality_weights": self.modality_weights,
            "top_evidence_count": self.top_evidence_count,
            "model_used": self.model_used,
            "is_fallback": self.is_fallback,
            "warning": self.warning,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Prompt builder
# ──────────────────────────────────────────────────────────────────────────────

def _build_context_block(evidence: List[FusedEvidenceResult]) -> str:
    """Serialize evidence chunks into a numbered, typed context block for the LLM."""
    lines = []
    for item in evidence:
        chunk = item.chunk
        label = chunk.citation_label

        if chunk.modality == ModalityType.TEXT:
            lines.append(f"[EVIDENCE {item.rank} | {label} | TEXT]\n{chunk.content.strip()}")
        elif chunk.modality == ModalityType.TABLE:
            lines.append(f"[EVIDENCE {item.rank} | {label} | TABLE]\n{chunk.content.strip()}")
        elif chunk.modality in (ModalityType.FIGURE, ModalityType.CHART):
            caption = chunk.metadata.get("caption") or chunk.content or ""
            lines.append(f"[EVIDENCE {item.rank} | {label} | FIGURE]\nCaption: {caption.strip()}")

    return "\n\n".join(lines)


def _build_prompt(query: str, context_block: str, intent: str) -> str:
    """Build the full grounding prompt for the LLM."""
    return textwrap.dedent(f"""
        You are a precise, citation-grounded document question-answering assistant.
        Your task is to answer the user's question based SOLELY on the evidence provided below.
        Do not use any prior knowledge, make assumptions, or introduce external facts.

        DETECTED QUERY INTENT: {intent}

        ── EVIDENCE CONTEXT ────────────────────────────────────────────────────
        {context_block}
        ────────────────────────────────────────────────────────────────────────

        INSTRUCTIONS:
        1. Answer the question in 2–5 clear, factual sentences.
        2. Every factual claim MUST be immediately followed by its citation in square
           brackets, exactly matching the label shown in the evidence header
           (e.g. [Page 3, Chunk 5], [Page 2, Table 1], [Page 5, Figure 2]).
        3. If multiple evidence items support a claim, include all relevant citations.
        4. If the answer cannot be determined from the evidence, respond with:
           "The provided document evidence is insufficient to answer this question."
        5. Do not speculate, hallucinate, or add information not present in the evidence.
        6. Do not restate the question in your answer.

        USER QUESTION: {query}

        ANSWER:
    """).strip()


# ──────────────────────────────────────────────────────────────────────────────
# Mock generator (no API key required)
# ──────────────────────────────────────────────────────────────────────────────

def _detect_list_intent(query: str) -> bool:
    """Detect if the query likely expects a list/steps answer."""
    list_keywords = [
        "list", "lists", "feature", "features", "step", "steps", "type", "types",
        "item", "items", "point", "points", "example", "examples", "procedure",
    ]
    q = query.lower()
    return any(kw in q for kw in list_keywords)


def _mock_answer(query: str, evidence: List[FusedEvidenceResult]) -> str:
    """Generate a deterministic structured mock answer for testing / demo purposes.
    Includes simple bullet formatting when list intent is detected.
    """
    if not evidence:
        return "The provided document evidence is insufficient to answer this question."

    is_list = _detect_list_intent(query)
    parts = []
    for i, item in enumerate(evidence[:3], start=1):
        chunk = item.chunk
        label = chunk.citation_label
        # For figures, prefer caption if content empty
        if not chunk.content and chunk.modality in (ModalityType.FIGURE, ModalityType.CHART):
            preview = chunk.metadata.get("caption", "[no caption]")
        else:
            preview = (chunk.content[:120] if chunk.content else "...").replace("\n", " ").strip()
        modality_tag = chunk.modality.value.capitalize()
        if is_list:
            parts.append(f"- {preview} ({label})")
        else:
            parts.append(
                f"Based on {modality_tag} evidence, the document states: \"{preview}...\" {label}."
            )
    if is_list:
        answer_body = "\n".join(parts)
    else:
        answer_body = " ".join(parts)
    return answer_body + (
        " (Note: This is a mock answer generated without an LLM API key. "
        "Provide a GEMINI_API_KEY in your .env file to enable real grounded generation.)"
    )


# ──────────────────────────────────────────────────────────────────────────────
# Main answer generator
# ──────────────────────────────────────────────────────────────────────────────

class AnswerGenerator:
    """Generates grounded, citation-enforced answers from fused multimodal evidence.

    Supports:
    - Gemini 1.5 Flash (via google-generativeai SDK) when GEMINI_API_KEY is set.
    - A deterministic mock fallback for testing when no API key is available.

    Args:
        model_name:  Gemini model to use (default from config).
        temperature: LLM temperature (lower = more factual).
        max_tokens:  Maximum tokens for the LLM response.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ):
        self.model_name = model_name or AppConfig.LLM_MODEL
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._gemini_model = None
        self._api_available = False

        api_key = AppConfig.GEMINI_API_KEY
        if api_key:
            try:
                import google.generativeai as genai
                genai.configure(api_key=api_key)
                self._gemini_model = genai.GenerativeModel(
                    model_name=self.model_name,
                    generation_config={
                        "temperature": self.temperature,
                        "max_output_tokens": self.max_tokens,
                    },
                )
                self._api_available = True
            except Exception:
                self._api_available = False

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def generate(
        self,
        query: str,
        evidence: List[FusedEvidenceResult],
        analysis: QueryAnalysisResult,
    ) -> AnswerResult:
        """Generate a grounded answer with inline citations.

        Args:
            query:    Original user question.
            evidence: Fused, ranked evidence from EvidenceFuser.
            analysis: Query analysis result (intent + weights).

        Returns:
            AnswerResult with the grounded answer text and structured citations.
        """
        if not evidence:
            return AnswerResult(
                query=query,
                answer="The provided document evidence is insufficient to answer this question.",
                citations=[],
                intent=analysis.detected_intent,
                modality_weights=self._serialize_weights(analysis),
                top_evidence_count=0,
                model_used="none",
                warning="No evidence chunks were retrieved for this query.",
            )

        context_block = _build_context_block(evidence)
        intent = analysis.detected_intent

        # Generate answer
        if self._api_available and self._gemini_model is not None:
            answer_text, model_used, warning = self._call_gemini(query, context_block, intent)
        else:
            answer_text = _mock_answer(query, evidence)
            model_used = "mock_generator"
            warning = (
                "GEMINI_API_KEY not set or Gemini SDK unavailable. "
                "Using mock answer generation. Add your key to .env to enable real LLM answers."
            )

        # Build citation list from evidence used
        citations = self._build_citations(evidence)

        return AnswerResult(
            query=query,
            answer=answer_text,
            citations=citations,
            intent=intent,
            modality_weights=self._serialize_weights(analysis),
            top_evidence_count=len(evidence),
            model_used=model_used,
            is_fallback=(model_used == "mock_generator"),
            warning=warning,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _call_gemini(
        self,
        query: str,
        context_block: str,
        intent: str,
    ):
        """Call the Gemini API and return (answer_text, model_id, warning)."""
        try:
            prompt = _build_prompt(query, context_block, intent)
            response = self._gemini_model.generate_content(prompt)
            answer_text = response.text.strip() if response.text else ""
            if not answer_text:
                return (
                    "The model returned an empty response. Please try rephrasing your question.",
                    self.model_name,
                    "Empty LLM response received.",
                )
            return answer_text, self.model_name, None
        except Exception as exc:
            return (
                f"An error occurred during answer generation: {exc}",
                self.model_name,
                f"LLM call failed: {exc}",
            )

    @staticmethod
    def _build_citations(evidence: List[FusedEvidenceResult]) -> List[Citation]:
        """Build a deduplicated list of Citation objects from evidence."""
        seen_ids = set()
        citations = []
        for item in evidence:
            chunk = item.chunk
            if chunk.chunk_id in seen_ids:
                continue
            seen_ids.add(chunk.chunk_id)
            citations.append(
                Citation(
                    chunk_id=chunk.chunk_id,
                    citation_label=chunk.citation_label,
                    modality=chunk.modality.value,
                    page_number=chunk.page_number,
                    content_preview=chunk.content[:200] if chunk.content else "",
                )
            )
        return citations

    @staticmethod
    def _serialize_weights(analysis: QueryAnalysisResult) -> Dict[str, float]:
        """Convert ModalityType-keyed weights to string-keyed for serialization."""
        return {
            m.value if isinstance(m, ModalityType) else str(m): round(float(w), 4)
            for m, w in analysis.modality_weights.items()
        }
