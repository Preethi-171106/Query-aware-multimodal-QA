"""Query Understanding and Modality Selection Engine.

Analyzes user questions to determine semantic intent and computes normalized
modality weights (TEXT, TABLE, FIGURE) using rule-based signal detection.

Note: This component is a transparent, configurable rule-based baseline router
designed for deterministic evaluation, not a trained neural classifier.
"""

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from src.indexing.schema import ModalityType


@dataclass
class QueryAnalysisResult:
    """Encapsulates the structured output of query intent and modality routing."""
    query: str
    detected_intent: str
    modality_weights: Dict[ModalityType, float]
    explanation: str
    confidence: float
    routing_method: str = "Rule-based signal routing (heuristic baseline)"
    signals_detected: Dict[str, List[str]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert analysis result to a serializable dictionary."""
        return {
            "query": self.query,
            "detected_intent": self.detected_intent,
            "modality_weights": {
                m.value if isinstance(m, ModalityType) else str(m): round(float(w), 4)
                for m, w in self.modality_weights.items()
            },
            "explanation": self.explanation,
            "confidence": round(float(self.confidence), 3),
            "routing_method": self.routing_method,
            "signals_detected": self.signals_detected,
        }


class QueryAnalyzer:
    """Analyzes questions and predicts modality affinity weights for evidence retrieval."""

    # Explicit lexical signals by modality
    TABLE_STRONG_SIGNALS = [
        "table", "tabular", "rows", "columns", "spreadsheet", "grid"
    ]
    TABLE_NUMERIC_SIGNALS = [
        "highest", "lowest", "maximum", "minimum", "compare", "comparison",
        "percentage", "percent", "growth rate", "revenue", "ebitda", "net income",
        "margin", "ratio", "metrics", "f1-score", "accuracy", "benchmark",
        "average", "total", "sum", "decrease", "increase", "difference"
    ]

    FIGURE_STRONG_SIGNALS = [
        "figure", "fig", "chart", "diagram", "plot", "graph", "illustration",
        "flowchart", "bar chart", "line chart", "pie chart", "histogram",
        "scatterplot", "architecture diagram", "visualization"
    ]
    FIGURE_VISUAL_SIGNALS = [
        "trend", "trendline", "curve", "convergence", "pattern", "depicted",
        "shown in", "according to the diagram", "visual", "axis", "plotted"
    ]

    TEXT_STRONG_SIGNALS = [
        "define", "definition", "explain", "explanation", "describe", "description",
        "concept", "principle", "why", "how does", "what is", "background",
        "motivation", "summary", "summarize", "overview", "policy", "conclude",
        "purpose", "meaning"
    ]

    def __init__(
        self,
        default_weights: Optional[Dict[ModalityType, float]] = None
    ):
        # Baseline prior distribution when no explicit signals exist
        self.default_weights = default_weights or {
            ModalityType.TEXT: 0.40,
            ModalityType.TABLE: 0.30,
            ModalityType.FIGURE: 0.30,
        }
        self._validate_weights(self.default_weights)

    @staticmethod
    def _validate_weights(weights: Dict[ModalityType, float]) -> None:
        """Ensure weights are non-negative, finite, and normalize to 1.0."""
        for m, w in weights.items():
            if not isinstance(w, (int, float)) or w < 0 or not (w == w):  # check nan
                raise ValueError(f"Invalid modality weight for {m}: {w}")

    @staticmethod
    def _normalize(weights: Dict[ModalityType, float]) -> Dict[ModalityType, float]:
        """Normalize a dictionary of raw weights to strictly sum to 1.0."""
        total = sum(weights.values())
        if total <= 1e-9:
            return {
                ModalityType.TEXT: 0.40,
                ModalityType.TABLE: 0.30,
                ModalityType.FIGURE: 0.30,
            }
        return {m: float(w / total) for m, w in weights.items()}

    def analyze(self, query: str) -> QueryAnalysisResult:
        """Analyze a user query and determine intent and normalized modality weights."""
        if not query or not query.strip():
            return QueryAnalysisResult(
                query=query or "",
                detected_intent="Empty / Fallback",
                modality_weights=self.default_weights.copy(),
                explanation="Default fallback prior applied because the query was empty.",
                confidence=0.0,
                signals_detected={"text": [], "table": [], "figure": []}
            )

        q_lower = query.lower().strip()
        words = set(re.findall(r'[a-zA-Z0-9]+(?:[-_.][a-zA-Z0-9]+)*', q_lower))

        detected_signals = {
            "table": [],
            "figure": [],
            "text": []
        }

        # 1. Detect Table Signals
        table_score = 0.0
        for s in self.TABLE_STRONG_SIGNALS:
            if s in q_lower:
                table_score += 3.5
                detected_signals["table"].append(s)
        for s in self.TABLE_NUMERIC_SIGNALS:
            if s in q_lower or s in words:
                table_score += 1.5
                detected_signals["table"].append(s)
        # Check for numeric quantities (digits / percentages / years)
        numeric_matches = re.findall(r'\b\d+(?:\.\d+)?%?\b', q_lower)
        if numeric_matches:
            table_score += min(len(numeric_matches) * 0.8, 2.4)
            detected_signals["table"].extend(numeric_matches[:3])

        # 2. Detect Figure Signals
        figure_score = 0.0
        for s in self.FIGURE_STRONG_SIGNALS:
            if s in q_lower:
                figure_score += 3.5
                detected_signals["figure"].append(s)
        for s in self.FIGURE_VISUAL_SIGNALS:
            if s in q_lower:
                figure_score += 1.5
                detected_signals["figure"].append(s)

        # 3. Detect Text / Conceptual Signals
        text_score = 0.0
        for s in self.TEXT_STRONG_SIGNALS:
            if s in q_lower or s in words:
                text_score += 2.0
                detected_signals["text"].append(s)

        # Baseline priors so no modality is completely zeroed out
        raw_weights = {
            ModalityType.TEXT: 1.0 + text_score,
            ModalityType.TABLE: 0.6 + table_score,
            ModalityType.FIGURE: 0.6 + figure_score,
        }

        normalized_weights = self._normalize(raw_weights)

        # Classify Primary Intent & Construct Explanation
        table_w = normalized_weights[ModalityType.TABLE]
        fig_w = normalized_weights[ModalityType.FIGURE]
        text_w = normalized_weights[ModalityType.TEXT]

        has_table_signals = bool(detected_signals["table"])
        has_fig_signals = bool(detected_signals["figure"])
        has_text_signals = bool(detected_signals["text"])

        # Determine Intent
        if fig_w >= 0.45 and (fig_w > table_w + 0.1):
            intent = "Chart / Figure Interpretation"
            explanation = (
                f"Assigned high figure weight ({fig_w:.1%}) due to visual/chart indicators: "
                f"{', '.join(detected_signals['figure'][:4])}."
            )
            confidence = min(0.60 + fig_w * 0.35, 0.95)

        elif table_w >= 0.45 and (table_w > fig_w + 0.1):
            if any(s in detected_signals["table"] for s in self.TABLE_STRONG_SIGNALS):
                intent = "Table Lookup"
            else:
                intent = "Numeric Comparison / Statistics"
            explanation = (
                f"Assigned high table weight ({table_w:.1%}) due to numerical/tabular indicators: "
                f"{', '.join(detected_signals['table'][:4])}."
            )
            confidence = min(0.60 + table_w * 0.35, 0.95)

        elif has_table_signals and has_fig_signals:
            intent = "Mixed Modality"
            explanation = (
                f"Detected multi-modal cross-over signals across tables ({', '.join(detected_signals['table'][:2])}) "
                f"and figures ({', '.join(detected_signals['figure'][:2])})."
            )
            confidence = 0.75

        elif has_text_signals and text_w >= 0.45:
            intent = "General Factual / Text"
            explanation = (
                f"Assigned high text weight ({text_w:.1%}) due to conceptual/explanatory terms: "
                f"{', '.join(detected_signals['text'][:4])}."
            )
            confidence = min(0.60 + text_w * 0.35, 0.95)

        else:
            intent = "General Factual / Text (Default Fallback)"
            explanation = (
                "No strong modality-specific cues detected. Applied balanced general prior with "
                "slight preference for comprehensive text."
            )
            confidence = 0.50

        return QueryAnalysisResult(
            query=query,
            detected_intent=intent,
            modality_weights=normalized_weights,
            explanation=explanation,
            confidence=confidence,
            signals_detected=detected_signals
        )
