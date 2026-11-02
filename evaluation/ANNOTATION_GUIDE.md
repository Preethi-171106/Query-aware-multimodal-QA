# ANNOTATION GUIDE – Query‑Aware Multimodal Evidence Fusion

## Purpose
This guide explains how to label the relevance of evidence chunks (text, tables, figures) for each benchmark question. The resulting CSV (`ground_truth.csv`) will be used **only** for metric calculation; it is never generated automatically.

## Columns in `annotation_template.csv`
| Column | Description |
|--------|-------------|
| `question_id` | Integer index matching the order in `sample_questions.json` |
| `question_text` | The exact benchmark question string |
| `chunk_id` | Identifier of the indexed chunk (e.g., `doc_af88fad1358a_p10_c3`) |
| `page_number` | PDF page where the chunk originates |
| `modality` | One of `TEXT`, `TABLE`, `FIGURE` |
| `chunk_text` | Full text of the chunk **or** the caption/contents for tables/figures |
| `relevance_label` | **Leave blank** until a human annotator decides (values: `relevant` / `non‑relevant` or any custom label) |
| `annotation_notes` | Optional free‑form notes for the annotator |

## Relevance Judgment Rules
1. **Text chunks** – A chunk is *relevant* if it directly answers the question or provides essential supporting information. Partial coverage is still considered relevant.
2. **Table chunks** – Consider the table as a whole. If the table contains the data needed to answer the question (even if you would need to read a specific cell), mark it *relevant*.
3. **Figure chunks** – Use the figure caption and any accompanying description. If the visual conveys the answer (e.g., a diagram showing a process), label it *relevant*.
4. **No guesswork** – If you are unsure, leave the label blank and add a note. The evaluation script will treat missing labels as *no ground‑truth* and report metrics as `N/A`.

## Workflow
1. Open `evaluation/annotation_template.csv` in a spreadsheet editor that preserves line breaks (e.g., Excel, LibreOffice, or a plain‑text editor with CSV support).
2. For each row, read the `chunk_text` column and decide whether it answers the `question_text`.
3. Fill the `relevance_label` column with `relevant` or `non‑relevant` (or leave blank).
4. Optionally add comments in `annotation_notes`.
5. Save the file **as** `ground_truth.csv` in the same `evaluation/` directory.

## Important
- Do **not** modify `annotation_template.csv`; it is the source of candidates.
- Do **not** edit any other project files while annotating.
- The evaluation script will only compute Recall@K and MRR for rows where `relevance_label` is provided.
