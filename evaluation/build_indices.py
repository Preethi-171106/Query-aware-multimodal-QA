# -*- coding: utf-8 -*-
"""Build FAISS and BM25 indices for the uploaded PDF.

This script performs the full ingestion pipeline used by the app and the
evaluation suite:

1. Load the PDF from ``data/uploads`` using :class:`PDFLoader`.
2. Extract multimodal evidence (text, tables, figures) via :class:`TextExtractor`.
3. Generate dense embeddings for each evidence chunk with :class:`EmbeddingManager`.
4. Populate a :class:`FAISSVectorStore` with the embeddings and save the index.
5. Index the raw evidence chunks with a :class:`LexicalRetriever` and persist the
   lexical index metadata.

The produced files are written under ``data/indices`` and are later consumed by
``evaluation.evaluate``.
"""

import json
import sys
from pathlib import Path

import sys, os
# Ensure project root is in PYTHONPATH for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.ingestion.pdf_loader import PDFLoader
from src.ingestion.text_extractor import TextExtractor
from src.indexing.embedding_manager import EmbeddingManager
from src.indexing.vector_store import FAISSVectorStore
from src.retrieval.lexical_retriever import LexicalRetriever

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
UPLOADS_DIR = Path(__file__).parents[1] / "data" / "uploads"
INDICES_DIR = Path(__file__).parents[1] / "data" / "indices"
PDF_NAME = "java_tutorial.pdf"  # Expected PDF used throughout the project

def main() -> int:
    """Execute the build pipeline.

    Returns
    -------
    int
        Exit status compatible with ``sys.exit`` – ``0`` on success.
    """
    pdf_path = UPLOADS_DIR / PDF_NAME
    if not pdf_path.is_file():
        print(f"[ERROR] Expected PDF not found: {pdf_path}")
        return 1

    # ---------------------------------------------------------------
    # 1. Load / validate the PDF
    # ---------------------------------------------------------------
    loader = PDFLoader()
    try:
        doc_info = loader.process_and_save(pdf_path)
    except Exception as exc:
        print(f"[ERROR] PDF loading failed: {exc}")
        return 1

    print("[INFO] PDF loaded and stored:")
    print(f"   ID: {doc_info.document_id}")
    print(f"   Pages: {doc_info.page_count}")

    # ---------------------------------------------------------------
    # 2. Extract multimodal evidence
    # ---------------------------------------------------------------
    extractor = TextExtractor()
    extracted_doc = extractor.extract_document(
        doc_info=doc_info,
        extract_tables=True,
        extract_figures=True,
    )
    print(f"[INFO] Extracted {len(extracted_doc.chunks)} evidence chunks.")
    print(f"   Text: {extracted_doc.total_text_chunks}, Tables: {extracted_doc.total_tables}, Figures: {extracted_doc.total_figures}")

    # ---------------------------------------------------------------
    # 3. Generate dense embeddings
    # ---------------------------------------------------------------
    embed_mgr = EmbeddingManager(mock_dim=384)
    embeddings, valid_chunks = embed_mgr.embed_chunks(extracted_doc.chunks)
    if embeddings.shape[0] == 0:
        print("[WARN] No embeddings were generated – nothing to index.")
        return 1
    print(f"[INFO] Generated {embeddings.shape[0]} embeddings (dim={embed_mgr.dimension}).")

    # ---------------------------------------------------------------
    # 4. Build and persist FAISS index
    # ---------------------------------------------------------------
    vstore = FAISSVectorStore(dimension=embed_mgr.dimension)
    added = vstore.add_chunks(valid_chunks, embeddings)
    print(f"[INFO] Added {added} chunks to FAISS index (total vectors={vstore.total_vectors}).")
    index_path, meta_path = vstore.save(INDICES_DIR)
    print(f"[INFO] FAISS index saved to {index_path}\n   metadata saved to {meta_path}")

    # ---------------------------------------------------------------
    # 5. Build and persist lexical (BM25) index
    # ---------------------------------------------------------------
    lexical = LexicalRetriever()
    lexical.index_chunks(extracted_doc.chunks)
    lexical_meta_path = INDICES_DIR / "lexical_chunks.json"
    lexical_meta_path.write_text(
        json.dumps([c.to_dict() for c in extracted_doc.chunks], indent=2),
        encoding="utf-8",
    )
    print(f"[INFO] Lexical index metadata saved to {lexical_meta_path}")

    print("\n[INFO] Index building complete.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
