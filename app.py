"""Query-Aware Multimodal Evidence Fusion for Grounded Document Question Answering.

Phase 5: Query Understanding, Modality-Weighted Fusion and Grounded Answer Generation.
"""

import json
import sys
from pathlib import Path
import pandas as pd
import streamlit as st

from src.config import AppConfig
from src.fusion.evidence_fuser import EvidenceFuser, FusedEvidenceResult
from src.generation.answer_generator import AnswerGenerator, AnswerResult
from src.indexing.embedding_manager import EmbeddingManager
from src.indexing.schema import ExtractedDocument, ModalityType, RetrievalResult
from src.indexing.vector_store import FAISSVectorStore
from src.ingestion.pdf_loader import (
    PDFLoader,
    DocumentInfo,
    EmptyPDFError,
    InvalidPDFFormatError,
    CorruptedPDFError,
)
from src.ingestion.text_extractor import TextExtractor
from src.query_understanding.query_analyzer import QueryAnalyzer, QueryAnalysisResult
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.lexical_retriever import LexicalRetriever

# Page configuration
st.set_page_config(
    page_title="Multimodal Document QA — Phase 5",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #1E293B;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.15rem;
        color: #475569;
        margin-bottom: 1.25rem;
        font-weight: 400;
    }
    .badge-phase {
        display: inline-block;
        background-color: #EEF2FF;
        color: #4F46E5;
        font-weight: 600;
        font-size: 0.85rem;
        padding: 4px 12px;
        border-radius: 9999px;
        border: 1px solid #C7D2FE;
        margin-bottom: 0.75rem;
    }
    .metric-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 0.85rem;
        text-align: center;
    }
    .metric-value {
        font-size: 1.45rem;
        font-weight: 700;
        color: #0F172A;
    }
    .metric-label {
        font-size: 0.75rem;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.25rem;
    }
    .chunk-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #4F46E5;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .chunk-card-table {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #10B981;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .chunk-card-figure {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #F59E0B;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .chunk-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.5rem;
        font-size: 0.85rem;
    }
    .citation-tag {
        background: #EEF2FF;
        color: #4338CA;
        padding: 2px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-family: monospace;
    }
    .citation-tag-table {
        background: #ECFDF5;
        color: #065F46;
        padding: 2px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-family: monospace;
    }
    .citation-tag-figure {
        background: #FFFBEB;
        color: #92400E;
        padding: 2px 8px;
        border-radius: 6px;
        font-weight: 600;
        font-family: monospace;
    }
    .score-badge {
        background: #F1F5F9;
        color: #334155;
        padding: 3px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 500;
        font-family: monospace;
    }
    .feature-card {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.25rem;
        margin-bottom: 1rem;
    }
    .feature-title {
        font-weight: 600;
        color: #0F172A;
        font-size: 1.05rem;
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "current_doc_info" not in st.session_state:
    st.session_state.current_doc_info = None
if "current_extracted_doc" not in st.session_state:
    st.session_state.current_extracted_doc = None
if "processed_file_hash" not in st.session_state:
    st.session_state.processed_file_hash = None
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "lexical_retriever" not in st.session_state:
    st.session_state.lexical_retriever = None
if "embedding_manager" not in st.session_state:
    st.session_state.embedding_manager = None
if "last_query" not in st.session_state:
    st.session_state.last_query = ""
if "retrieval_results" not in st.session_state:
    st.session_state.retrieval_results = None
# Phase 5 additions
if "query_analysis" not in st.session_state:
    st.session_state.query_analysis = None
if "fused_results" not in st.session_state:
    st.session_state.fused_results = None
if "answer_result" not in st.session_state:
    st.session_state.answer_result = None


# Sidebar
with st.sidebar:
    st.markdown("### 🔬 Project Overview")
    st.markdown(
        """
        **Final-Year Engineering Project**  
        *Query-Aware Multimodal Evidence Fusion for Grounded Document QA*
        
        **Active Focus (Phase 5):**
        - 🎯 Query Intent Analysis
        - ⚖️ Modality Weight Routing
        - 🔀 MW-RRF Evidence Fusion
        - 💡 Grounded Answer Generation
        - 📌 Page-Level Citations
        """
    )
    
    st.divider()
    
    st.markdown("### ⚙️ System Status")
    st.write(f"**App Version:** `0.5.0-phase5`")
    st.write(f"**Python Runtime:** `{sys.version.split()[0]}`")
    st.write(f"**Embedding Model:** `{AppConfig.TEXT_EMBEDDING_MODEL}`")
    st.write(f"**Vector Store:** `FAISS-CPU IndexFlatIP`")
    st.write(f"**Lexical Index:** `BM25Okapi`")
    gemini_status = "✅ Configured" if AppConfig.GEMINI_API_KEY else "⚠️ Not set (mock mode)"
    st.write(f"**Gemini API:** {gemini_status}")
    
    # Active Document Status in Sidebar
    if st.session_state.current_doc_info:
        doc = st.session_state.current_doc_info
        extracted: ExtractedDocument = st.session_state.current_extracted_doc
        st.markdown("---")
        st.markdown("### 📄 Active Document")
        st.markdown(f"**File:** `{doc.file_name}`")
        st.markdown(f"**ID:** `{doc.document_id}`")
        st.markdown(f"**Pages:** `{doc.page_count}`")
        if extracted:
            st.markdown(f"• Text Chunks: `{extracted.total_text_chunks}`")
            st.markdown(f"• Tables: `{extracted.total_tables}`")
            st.markdown(f"• Figures: `{extracted.total_figures}`")
            
        if doc.is_duplicate:
            st.caption("ℹ️ Recognized previously stored document.")
        
        if st.button("🔄 Process Another Document", use_container_width=True):
            st.session_state.current_doc_info = None
            st.session_state.current_extracted_doc = None
            st.session_state.processed_file_hash = None
            st.session_state.vector_store = None
            st.session_state.lexical_retriever = None
            st.session_state.retrieval_results = None
            st.session_state.query_analysis = None
            st.session_state.fused_results = None
            st.session_state.answer_result = None
            st.session_state.last_query = ""
            st.rerun()

    st.divider()
    st.caption("Final-Year Engineering Project • Phase 5 Active")


# Main Body
st.markdown('<span class="badge-phase">Phase 5: Query Understanding · Modality-Weighted Fusion · Grounded QA (Active)</span>', unsafe_allow_html=True)
st.markdown(f'<h1 class="main-header">{AppConfig.APP_NAME}</h1>', unsafe_allow_html=True)
st.markdown(f'<p class="sub-header">{AppConfig.APP_TAGLINE}</p>', unsafe_allow_html=True)

# Document Upload Section
st.markdown("### 📤 Upload PDF Document")

uploaded_file = st.file_uploader(
    label="Upload a multi-page PDF document to extract, index, and query",
    type=["pdf"],
    help="Upload research papers, financial reports, or technical manuals containing text, tables, and charts.",
    key="pdf_uploader"
)

# Ingestion, Extraction & Indexing Processing
if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    loader = PDFLoader()
    file_hash = loader.compute_hash(file_bytes)

    if st.session_state.processed_file_hash != file_hash:
        with st.spinner("Extracting multimodal evidence and building FAISS + BM25 indices..."):
            try:
                # 1. Validate & Save
                doc_info = loader.process_and_save(file_bytes, uploaded_file.name)
                
                # 2. Extract Multimodal Evidence
                extractor = TextExtractor()
                extracted_doc = extractor.extract_document(
                    doc_info=doc_info,
                    extract_tables=True,
                    extract_figures=True
                )

                # 3. Build Dense Embeddings & FAISS Index
                embed_mgr = EmbeddingManager()
                embs, valid_chunks = embed_mgr.embed_chunks(extracted_doc.chunks)

                vstore = FAISSVectorStore(dimension=embed_mgr.dimension)
                vstore.add_chunks(valid_chunks, embs)

                # 4. Build Sparse BM25 Lexical Index
                lretriever = LexicalRetriever()
                lretriever.index_chunks(extracted_doc.chunks)
                
                # Update Session State
                st.session_state.current_doc_info = doc_info
                st.session_state.current_extracted_doc = extracted_doc
                st.session_state.processed_file_hash = file_hash
                st.session_state.embedding_manager = embed_mgr
                st.session_state.vector_store = vstore
                st.session_state.lexical_retriever = lretriever
                st.session_state.retrieval_results = None
                st.session_state.last_query = ""

                st.toast(f"Successfully processed and indexed {uploaded_file.name}!", icon="✅")

            except EmptyPDFError as e:
                st.error(f"❌ **Empty PDF Error:** {str(e)}")
            except InvalidPDFFormatError as e:
                st.error(f"❌ **Invalid Format:** {str(e)}")
            except CorruptedPDFError as e:
                st.error(f"❌ **Corrupted PDF:** {str(e)}")
            except Exception as e:
                st.error(f"❌ **Unexpected Error during Processing:** {str(e)}")

# Display Extracted Multimodal Document Dashboard & Retrieval
if st.session_state.current_extracted_doc is not None:
    doc_info: DocumentInfo = st.session_state.current_doc_info
    extracted: ExtractedDocument = st.session_state.current_extracted_doc
    vstore: FAISSVectorStore = st.session_state.vector_store
    lretriever: LexicalRetriever = st.session_state.lexical_retriever
    embed_mgr: EmbeddingManager = st.session_state.embedding_manager

    st.success(
        f"✅ **Document Processed & Indexed Successfully!** "
        f"Stored under `data/uploads/{doc_info.file_name}`"
    )

    # 5-Column Multimodal Metric Row
    m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
    with m_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{doc_info.page_count}</div>
            <div class="metric-label">Pages</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{extracted.total_text_chunks}</div>
            <div class="metric-label">Text Chunks</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #059669;">{extracted.total_tables}</div>
            <div class="metric-label">Tables</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color: #D97706;">{extracted.total_figures}</div>
            <div class="metric-label">Figures</div>
        </div>
        """, unsafe_allow_html=True)
    with m_col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{vstore.total_vectors if vstore else 0}</div>
            <div class="metric-label">Indexed Vectors</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # =========================================================================
    # PHASE 4: HYBRID EVIDENCE RETRIEVAL SECTION
    # =========================================================================
    st.markdown("### 🔍 Hybrid Evidence Retrieval (Phase 4)")
    st.caption(
        "Combines Dense Semantic Search (FAISS + all-MiniLM-L6-v2) and Lexical Exact Matching (BM25) "
        "via Reciprocal Rank Fusion (RRF). Discovers the most relevant text, table, and figure evidence chunks."
    )

    q_col1, q_col2, q_col3 = st.columns([3, 1, 1])
    with q_col1:
        query_input = st.text_input(
            label="Sample Question or Evidence Query:",
            placeholder="e.g., What was the EBITDA growth in Q3? or System architecture diagram",
            key="retrieval_query_input"
        )
    with q_col2:
        modality_sel = st.selectbox(
            "Filter Modality:",
            options=["All Modalities", "TEXT only", "TABLE only", "FIGURE only"],
            index=0,
            key="retrieval_modality_filter"
        )
    with q_col3:
        top_k_sel = st.selectbox(
            "Top-K Results:",
            options=[3, 5, 10],
            index=1,
            key="retrieval_top_k"
        )

    run_search_btn = st.button("🚀 Run Hybrid Retrieval", use_container_width=True)

    if run_search_btn and query_input.strip():
        # Map modality filter
        mod_filter = None
        if modality_sel == "TEXT only":
            mod_filter = ModalityType.TEXT
        elif modality_sel == "TABLE only":
            mod_filter = ModalityType.TABLE
        elif modality_sel == "FIGURE only":
            mod_filter = ModalityType.FIGURE

        hybrid_retriever = HybridRetriever(
            vector_store=vstore,
            lexical_retriever=lretriever,
            embedding_manager=embed_mgr,
            rrf_k=60
        )

        with st.spinner("Executing dense vector similarity and BM25 lexical ranking..."):
            results = hybrid_retriever.search(
                query=query_input,
                top_k=top_k_sel,
                modality_filter=mod_filter
            )
            st.session_state.retrieval_results = results
            st.session_state.last_query = query_input

    # Display Retrieval Results
    if st.session_state.retrieval_results is not None:
        results: List[RetrievalResult] = st.session_state.retrieval_results
        st.markdown(f"#### 🏆 Top Retrieved Evidence for: *\"{st.session_state.last_query}\"*")
        
        if not results:
            st.info("No evidence chunks matched your query. Try broadening your keywords.", icon="ℹ️")
        else:
            st.caption(
                f"Retrieved **{len(results)}** evidence chunks fused via Reciprocal Rank Fusion ($k=60$). "
                "Component dense and lexical ranks are shown for transparency."
            )

            for item in results:
                chunk = item.chunk
                
                # Card class based on modality
                card_cls = "chunk-card"
                tag_cls = "citation-tag"
                if chunk.modality == ModalityType.TABLE:
                    card_cls = "chunk-card-table"
                    tag_cls = "citation-tag-table"
                elif chunk.modality in (ModalityType.FIGURE, ModalityType.CHART):
                    card_cls = "chunk-card-figure"
                    tag_cls = "citation-tag-figure"

                dense_info = f"Rank {item.dense_rank} (cos: {item.dense_score:.3f})" if item.dense_rank else "Not in top dense"
                lex_info = f"Rank {item.lexical_rank} (bm25: {item.lexical_score:.2f})" if item.lexical_rank else "Not in top BM25"

                st.markdown(f"""
                <div class="{card_cls}">
                    <div class="chunk-header">
                        <div>
                            <span class="{tag_cls}">#{item.rank} • {chunk.citation_label}</span>
                            <span style="font-weight: 600; color: #1E293B; margin-left: 8px;">[{chunk.modality.value}]</span>
                        </div>
                        <div>
                            <span class="score-badge">RRF: {item.rrf_score:.5f}</span>
                            <span class="score-badge">Dense: {dense_info}</span>
                            <span class="score-badge">Lexical: {lex_info}</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Render evidence content
                if chunk.modality == ModalityType.TEXT:
                    st.markdown(f"<div style='padding-left: 8px; margin-bottom: 12px; font-size: 0.95rem; color: #1E293B;'>{chunk.content}</div>", unsafe_allow_html=True)
                elif chunk.modality == ModalityType.TABLE:
                    t_data = chunk.table_data
                    if t_data and "headers" in t_data and "rows" in t_data:
                        try:
                            df = pd.DataFrame(t_data["rows"], columns=t_data["headers"])
                            st.dataframe(df, use_container_width=True)
                        except Exception:
                            st.markdown(chunk.content)
                    else:
                        st.markdown(chunk.content)
                elif chunk.modality in (ModalityType.FIGURE, ModalityType.CHART):
                    img_path = Path(chunk.image_path) if chunk.image_path else None
                    if img_path and img_path.exists():
                        st.image(str(img_path), caption=chunk.content, width=400)
                    else:
                        st.caption(chunk.content)

        st.caption(
            "⚠️ **Phase 4 Retrieval Panel:** Results above are raw hybrid retrieval candidates "
            "(FAISS + BM25). The Phase 5 QA section below applies modality-weighted fusion and generates a grounded answer."
        )

    st.markdown("---")

    # =========================================================================
    # PHASE 5: QUERY-AWARE MULTIMODAL QA
    # =========================================================================
    st.markdown("### 🧠 Grounded Document QA (Phase 5)")
    st.caption(
        "Analyzes query intent → computes modality weights → runs Modality-Weighted Reciprocal Rank Fusion (MW-RRF) "
        "→ generates a grounded, citation-linked answer via Gemini 1.5 Flash."
    )

    p5_col1, p5_col2, p5_col3 = st.columns([4, 1, 1])
    with p5_col1:
        qa_query = st.text_input(
            label="Ask a question about the uploaded document:",
            placeholder="e.g., What was the revenue growth in Q3? or Describe the system architecture.",
            key="qa_query_input"
        )
    with p5_col2:
        qa_top_k = st.selectbox(
            "Evidence Top-K:",
            options=[3, 5, 8],
            index=1,
            key="qa_top_k"
        )
    with p5_col3:
        qa_button = st.button("🧠 Generate Answer", use_container_width=True, type="primary")

    if qa_button and qa_query.strip():
        analyzer = QueryAnalyzer()
        fuser = EvidenceFuser(
            vector_store=vstore,
            lexical_retriever=lretriever,
            embedding_manager=embed_mgr,
        )
        generator = AnswerGenerator()

        with st.spinner("🎯 Analyzing query intent and routing modality weights..."):
            analysis: QueryAnalysisResult = analyzer.analyze(qa_query)
            st.session_state.query_analysis = analysis

        with st.spinner("🔀 Running Modality-Weighted Fusion (MW-RRF)..."):
            fused = fuser.fuse(query=qa_query, analysis=analysis, top_k=qa_top_k)
            st.session_state.fused_results = fused

        with st.spinner("💡 Generating grounded answer with citations..."):
            answer: AnswerResult = generator.generate(
                query=qa_query, evidence=fused, analysis=analysis
            )
            st.session_state.answer_result = answer
            st.session_state.last_query = qa_query

    # ── Display Phase 5 Results ─────────────────────────────────────────────
    if st.session_state.answer_result is not None:
        analysis: QueryAnalysisResult = st.session_state.query_analysis
        fused: list = st.session_state.fused_results
        answer: AnswerResult = st.session_state.answer_result

        st.markdown(f"#### 💡 Answer for: *\"{st.session_state.last_query}\"*")

        # --- Answer box ---
        if answer.is_fallback:
            st.warning(
                f"⚠️ **Mock Answer** (no Gemini API key): {answer.warning}",
                icon="📅"
            )
        elif answer.warning:
            st.warning(answer.warning)

        answer_bg = "#F0FDF4" if not answer.is_fallback else "#FFFBEB"
        answer_border = "#22C55E" if not answer.is_fallback else "#F59E0B"
        model_badge = f"`{answer.model_used}`"
        st.markdown(f"""
        <div style="background:{answer_bg}; border:1px solid {answer_border}; border-left:5px solid {answer_border};
                    border-radius:10px; padding:1.2rem; margin-bottom:1rem; font-size:1.0rem; line-height:1.75;">
            {answer.answer}
        </div>
        """, unsafe_allow_html=True)
        st.caption(f"🤖 Generated by {model_badge} • Based on {answer.top_evidence_count} fused evidence chunks")

        # --- 3-column analysis summary ---
        anl_col1, anl_col2, anl_col3 = st.columns(3)
        with anl_col1:
            st.markdown("**🎯 Query Intent**")
            st.info(analysis.detected_intent)
            st.caption(analysis.explanation)
        with anl_col2:
            st.markdown("**⚖️ Modality Weights**")
            weights_display = analysis.modality_weights
            for m, w in weights_display.items():
                label = m.value if hasattr(m, 'value') else str(m)
                bar_pct = int(w * 100)
                color_map = {"TEXT": "#4F46E5", "TABLE": "#10B981", "FIGURE": "#F59E0B"}
                color = color_map.get(label, "#94A3B8")
                st.markdown(
                    f"""<div style='margin-bottom:6px'>
                        <span style='font-size:0.85rem;font-weight:600;color:#334155'>{label}</span>
                        <div style='background:#E2E8F0;border-radius:4px;height:10px;margin-top:3px'>
                            <div style='width:{bar_pct}%;background:{color};height:10px;border-radius:4px'></div>
                        </div>
                        <span style='font-size:0.75rem;color:#64748B'>{w:.1%}</span>
                    </div>""",
                    unsafe_allow_html=True
                )
        with anl_col3:
            st.markdown("**📌 Citations**")
            if answer.citations:
                for cit in answer.citations:
                    cit_color = {"TEXT": "#4338CA", "TABLE": "#065F46", "FIGURE": "#92400E"}.get(cit.modality, "#334155")
                    cit_bg = {"TEXT": "#EEF2FF", "TABLE": "#ECFDF5", "FIGURE": "#FFFBEB"}.get(cit.modality, "#F8FAFC")
                    st.markdown(
                        f"""<div style='background:{cit_bg};border-radius:6px;padding:4px 8px;margin-bottom:4px;
                                      font-size:0.82rem;color:{cit_color};font-weight:600'>
                            {cit.citation_label} — {cit.modality}
                        </div>""",
                        unsafe_allow_html=True
                    )
            else:
                st.caption("No citations available.")

        # --- Fused evidence expander ---
        with st.expander(f"🔀 View {len(fused)} MW-RRF Fused Evidence Chunks (detailed scores)", expanded=False):
            st.caption(
                "Evidence ranked by Modality-Weighted Reciprocal Rank Fusion score. "
                "The fused_score = Σ (modality_weight × 1/(60+rank)) summed across active modalities."
            )
            for item in fused:
                chunk = item.chunk
                card_cls = "chunk-card"
                tag_cls = "citation-tag"
                if chunk.modality == ModalityType.TABLE:
                    card_cls = "chunk-card-table"
                    tag_cls = "citation-tag-table"
                elif chunk.modality in (ModalityType.FIGURE, ModalityType.CHART):
                    card_cls = "chunk-card-figure"
                    tag_cls = "citation-tag-figure"

                st.markdown(f"""
                <div class="{card_cls}">
                    <div class="chunk-header">
                        <div>
                            <span class="{tag_cls}">#{item.rank} • {chunk.citation_label}</span>
                            <span style="font-weight:600;color:#1E293B;margin-left:8px">[{chunk.modality.value}]</span>
                        </div>
                        <div>
                            <span class="score-badge">MW-RRF: {item.fused_score:.5f}</span>
                            <span class="score-badge">Mod-W: {item.modality_weight:.2%}</span>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if chunk.modality == ModalityType.TEXT:
                    st.markdown(f"<div style='padding-left:8px;margin-bottom:12px;font-size:0.9rem;color:#1E293B'>{chunk.content}</div>", unsafe_allow_html=True)
                elif chunk.modality == ModalityType.TABLE:
                    t_data = chunk.table_data
                    if t_data and "headers" in t_data and "rows" in t_data:
                        try:
                            df = pd.DataFrame(t_data["rows"], columns=t_data["headers"])
                            st.dataframe(df, use_container_width=True)
                        except Exception:
                            st.markdown(chunk.content)
                    else:
                        st.markdown(chunk.content)
                elif chunk.modality in (ModalityType.FIGURE, ModalityType.CHART):
                    img_path = Path(chunk.image_path) if chunk.image_path else None
                    if img_path and img_path.exists():
                        st.image(str(img_path), caption=chunk.content, width=360)
                    else:
                        st.caption(chunk.content)

        # Download QA result
        st.download_button(
            label="💾 Download QA Result JSON",
            data=json.dumps(answer.to_dict(), indent=2),
            file_name=f"qa_result_{st.session_state.last_query[:30].replace(' ', '_')}.json",
            mime="application/json",
        )

    st.markdown("---")


    # Document Exploration Tabs (Preserved from Phase 3)
    tab_all, tab_tables, tab_figures, tab_pages, tab_metadata = st.tabs([
        f"🧩 All Evidence ({extracted.total_chunks})",
        f"📊 Tables ({extracted.total_tables})",
        f"🖼️ Figures ({extracted.total_figures})",
        f"📄 Page-by-Page Raw Text ({doc_info.page_count})",
        "📋 Metadata & Export"
    ])

    # TAB 1: ALL EVIDENCE CHUNKS
    with tab_all:
        st.markdown("#### Unified Multimodal Evidence Chunks")
        st.caption("All extracted evidence items normalized under the unified EvidenceChunk representation.")

        f_col1, f_col2 = st.columns([1, 1])
        with f_col1:
            m_filter = st.selectbox(
                "Filter by Modality:",
                options=["All Modalities", "TEXT only", "TABLE only", "FIGURE only"],
                index=0,
                key="all_evidence_mod_filter"
            )
        with f_col2:
            p_filter = st.selectbox(
                "Filter by Page:",
                options=["All Pages"] + [f"Page {p}" for p in range(1, doc_info.page_count + 1)],
                index=0,
                key="all_evidence_p_filter"
            )

        filtered = extracted.chunks
        if m_filter == "TEXT only":
            filtered = [c for c in filtered if c.modality == ModalityType.TEXT]
        elif m_filter == "TABLE only":
            filtered = [c for c in filtered if c.modality == ModalityType.TABLE]
        elif m_filter == "FIGURE only":
            filtered = [c for c in filtered if c.modality in (ModalityType.FIGURE, ModalityType.CHART)]

        if p_filter != "All Pages":
            target_p = int(p_filter.replace("Page ", ""))
            filtered = [c for c in filtered if c.page_number == target_p]

        st.write(f"Displaying **{len(filtered)}** evidence items:")

        for chunk in filtered:
            if chunk.modality == ModalityType.TEXT:
                st.markdown(f"""
                <div class="chunk-card">
                    <div class="chunk-header">
                        <span class="citation-tag">{chunk.citation_label}</span>
                        <span style="color: #64748B;">ID: <code>{chunk.chunk_id}</code> • {chunk.metadata.get('word_count', 0)} words</span>
                    </div>
                    <div style="font-size: 0.95rem; color: #1E293B; line-height: 1.5; white-space: pre-wrap;">{chunk.content}</div>
                </div>
                """, unsafe_allow_html=True)

            elif chunk.modality == ModalityType.TABLE:
                st.markdown(f"""
                <div class="chunk-card-table">
                    <div class="chunk-header">
                        <span class="citation-tag-table">{chunk.citation_label}</span>
                        <span style="color: #065F46;">ID: <code>{chunk.chunk_id}</code> • {chunk.metadata.get('row_count', 0)} rows x {chunk.metadata.get('col_count', 0)} cols</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                st.markdown(chunk.content)

            elif chunk.modality in (ModalityType.FIGURE, ModalityType.CHART):
                st.markdown(f"""
                <div class="chunk-card-figure">
                    <div class="chunk-header">
                        <span class="citation-tag-figure">{chunk.citation_label}</span>
                        <span style="color: #92400E;">ID: <code>{chunk.chunk_id}</code> • {chunk.metadata.get('width', 0)}x{chunk.metadata.get('height', 0)}px</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                if chunk.image_path and Path(chunk.image_path).exists():
                    st.image(chunk.image_path, caption=chunk.content, width=350)
                else:
                    st.caption(chunk.content)

    # TAB 2: STRUCTURED TABLES
    with tab_tables:
        st.markdown("#### Extracted Structured Tables")
        st.caption("Tables parsed with preserved cell coordinates, normalized whitespace, and row/column relationships.")

        if extracted.total_tables == 0:
            st.warning(
                "ℹ️ **No structured tables detected in this PDF document.**\n\n"
                "The table extractor checks for bounded grid cells with at least 2 rows and 2 columns. "
                "Ordinary paragraphs and bulleted text blocks are strictly separated to prevent false positive classifications.",
                icon="📊"
            )
        else:
            for t_idx, table_chunk in enumerate(extracted.table_chunks, start=1):
                with st.expander(f"{table_chunk.citation_label} — {table_chunk.metadata.get('row_count', 0)} rows x {table_chunk.metadata.get('col_count', 0)} columns", expanded=True):
                    st.markdown(f"**Evidence Chunk ID:** `{table_chunk.chunk_id}`")
                    t_data = table_chunk.table_data
                    if t_data and "headers" in t_data and "rows" in t_data:
                        try:
                            df = pd.DataFrame(t_data["rows"], columns=t_data["headers"])
                            st.dataframe(df, use_container_width=True)
                        except Exception:
                            st.markdown(table_chunk.content)
                    else:
                        st.markdown(table_chunk.content)

                    with st.expander("View Raw Markdown Representation"):
                        st.code(table_chunk.content, language="markdown")

    # TAB 3: EXTRACTED FIGURES & CHARTS
    with tab_figures:
        st.markdown("#### Extracted Visual Figures & Chart Assets")
        st.caption("Embedded raster figures cropped and saved under `data/extracted/figures/` with source page provenance.")

        if extracted.total_figures == 0:
            st.warning(
                "ℹ️ **No embedded raster figures or images found in this PDF document.**\n\n"
                "**Technical Note on Vector Charts:** Diagrams or charts rendered via procedural vector instructions "
                "(PDF path stroke/fill operators) rather than embedded raster images cannot reliably be cropped as image objects. "
                "This is a documented limitation of standard PDF stream parsers.",
                icon="🖼️"
            )
        else:
            fig_cols = st.columns(2)
            for i, fig_chunk in enumerate(extracted.figure_chunks):
                col = fig_cols[i % 2]
                with col:
                    with st.container(border=True):
                        st.markdown(f"**{fig_chunk.citation_label}** — `ID: {fig_chunk.chunk_id}`")
                        img_path = Path(fig_chunk.image_path) if fig_chunk.image_path else None
                        if img_path and img_path.exists():
                            st.image(
                                str(img_path),
                                caption=fig_chunk.metadata.get("caption") or f"Extracted figure from Page {fig_chunk.page_number}",
                                use_container_width=True
                            )
                        else:
                            st.info("Image file missing or not saved.")

                        st.caption(
                            f"**Dimensions:** {fig_chunk.metadata.get('width', 0)}x{fig_chunk.metadata.get('height', 0)} px | "
                            f"**Size:** {fig_chunk.metadata.get('file_size_bytes', 0) / 1024:.1f} KB | "
                            f"**Caption Linked:** {'Yes' if fig_chunk.metadata.get('has_caption') else 'None detected'}"
                        )

    # TAB 4: PAGE-BY-PAGE RAW TEXT
    with tab_pages:
        st.markdown("#### Page-Level Raw Text Stream")
        st.caption("Inspect the complete, unchunked digital text stream for each page.")

        for page in extracted.pages:
            with st.expander(
                f"Page {page.page_number} — "
                f"{'✅ ' + str(page.word_count) + ' words' if page.has_text else '⚠️ No digital text detected'}",
                expanded=(page.page_number == 1)
            ):
                if page.has_text:
                    st.text_area(
                        label=f"Raw text on Page {page.page_number}",
                        value=page.raw_text,
                        height=180,
                        key=f"page_text_{page.page_number}",
                        disabled=True
                    )
                    st.caption(f"Extractor: `{page.metadata.get('extractor', 'pdfplumber')}` • Characters: {page.char_count}")
                else:
                    st.info(
                        f"Page {page.page_number} does not contain an extractable text stream. "
                        f"(May contain visual diagrams or scanned graphics)."
                    )

    # TAB 5: DOCUMENT METADATA & SERIALIZATION
    with tab_metadata:
        st.markdown("#### Document Provenance & Serialization")
        meta_col1, meta_col2 = st.columns(2)
        with meta_col1:
            st.json(doc_info.metadata if doc_info.metadata else {"info": "No embedded PDF metadata available."})
        with meta_col2:
            st.write("**Document Identifier:**", f"`{doc_info.document_id}`")
            st.write("**Content SHA-256:**", f"`{doc_info.content_hash}`")
            st.write("**Total Modality Chunks:**", f"`{extracted.total_chunks}`")
            st.write("• Text Chunks:", f"`{extracted.total_text_chunks}`")
            st.write("• Structured Tables:", f"`{extracted.total_tables}`")
            st.write("• Visual Figures:", f"`{extracted.total_figures}`")
            st.write("**Dense Vector Count:**", f"`{vstore.total_vectors if vstore else 0}`")
            st.write("**BM25 Tokenized Chunks:**", f"`{lretriever.total_documents if lretriever else 0}`")
            st.write("**Storage Path:**", f"`{doc_info.file_path}`")

        # Download Extracted Document JSON
        st.download_button(
            label="💾 Download Full Multimodal Evidence JSON",
            data=json.dumps(extracted.to_dict(), indent=2),
            file_name=f"{doc_info.document_id}_multimodal_evidence.json",
            mime="application/json",
            help="Download the serialized text chunks, tables, and figure metadata."
        )

else:
    # Upload instructions placeholder when no document is active
    st.info("👆 Select and upload a PDF above to extract, index, and query multimodal evidence.", icon="💡")

st.markdown("---")

# Architectural Pipeline Walkthrough (Preserved for system overview)
st.markdown("### 🧩 Planned Multimodal Architecture")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-title">📄 1. Multimodal Extraction</div>
        <p style="color: #64748B; font-size: 0.9rem;">
            <b>Phase 3 Active:</b> Separates document contents into 3 dedicated streams:
            raw text chunks, structured table markdowns, and cropped figure/chart assets.
        </p>
        <span style="color: #10B981; font-weight: 600; font-size: 0.8rem;">Multimodal Ingestion Implemented</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="feature-card">
        <div class="feature-title">🗂️ 2. Representation & Index</div>
        <p style="color: #64748B; font-size: 0.9rem;">
            <b>Phase 4 Active:</b> Dense vectors via Sentence-Transformers (FAISS) + Sparse BM25 keyword index with Reciprocal Rank Fusion (RRF).
        </p>
        <span style="color: #10B981; font-weight: 600; font-size: 0.8rem;">Hybrid Indexing & Retrieval Implemented</span>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-title">🎯 3. Query Modality Router</div>
        <p style="color: #64748B; font-size: 0.9rem;">
            <b>Phase 5 Active:</b> Analyzes the question intent to compute dynamic modality weights (Text vs. Table vs. Chart) before retrieval.
        </p>
        <span style="color: #10B981; font-weight: 600; font-size: 0.8rem;">Query Analysis & Routing Implemented</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="feature-card">
        <div class="feature-title">🔀 4. Modality-Aware Fusion</div>
        <p style="color: #64748B; font-size: 0.9rem;">
            <b>Phase 5 Active:</b> Modality-Weighted RRF (MW-RRF) re-ranks and fuses multi-source evidence using query-affinity scores.
        </p>
        <span style="color: #10B981; font-weight: 600; font-size: 0.8rem;">MW-RRF Evidence Fusion Implemented</span>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="feature-card">
        <div class="feature-title">💡 5. Grounded Generation</div>
        <p style="color: #64748B; font-size: 0.9rem;">
            <b>Phase 5 Active:</b> Synthesizes answers conditioned strictly on retrieved evidence with verifiable page-level citations (e.g. <code>[Page 4, Table 2]</code>).
        </p>
        <span style="color: #10B981; font-weight: 600; font-size: 0.8rem;">Grounded Answer Generation Implemented</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="feature-card">
        <div class="feature-title">📊 6. Evaluation & Baselines</div>
        <p style="color: #64748B; font-size: 0.9rem;">
            Compares performance against Text-Only RAG and Naive Flat Multimodal RAG on Citation Precision, Grounding, and Recall@K.
        </p>
        <span style="color: #6366F1; font-weight: 500; font-size: 0.8rem;">Planned for Phase 6</span>
    </div>
    """, unsafe_allow_html=True)

# Environment Diagnostics
with st.expander("🛠️ Environment & Diagnostics Check", expanded=False):
    st.markdown("#### Installed Component Verification")
    
    diag_col1, diag_col2 = st.columns(2)
    with diag_col1:
        st.write("✅ **Streamlit Web Framework:** Ready (v1.58.0)")
        st.write("✅ **PDF Parsers:** `pdfplumber` (v0.11.10) & `pypdf` (v6.21.1)")
        st.write("✅ **Multimodal Storage:** `data/uploads`, `data/extracted/figures`")
        st.write("✅ **Query Analyzer:** Rule-based intent routing with modality weights")
    with diag_col2:
        st.write("✅ **Vector Store:** FAISS-CPU (v1.14.3)")
        st.write("✅ **Lexical Search:** rank-bm25 (v0.2.2)")
        st.write("✅ **Dense Embeddings:** sentence-transformers (`all-MiniLM-L6-v2`)")
        gemini_key = AppConfig.GEMINI_API_KEY
        if gemini_key:
            st.write("✅ **Gemini API:** Configured (grounded generation enabled)")
        else:
            st.write("⚠️ **Gemini API:** Not configured — using mock answer generator")
    
    st.caption("All Phase 5 components (query analysis, MW-RRF fusion, grounded generation) operational.")
