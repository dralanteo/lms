# Text Analysis Workflows

A collection of independent Python workflows for topic modeling, long-text emotion classification, and LLM-assisted thematic analysis. The repository also includes a standalone HTML report for source-quote review.

Each workflow has its own dependencies and README so it can be installed and run independently.

## Included workflows

### Chunked BERTopic Pipeline

Location: [`bertopic-chunked-pipeline/`](bertopic-chunked-pipeline/)

Fits BERTopic on an initial data chunk and assigns topics to the remaining records using the fitted model.

Key capabilities:

- CSV and Parquet input
- Duplicate removal and text cleaning
- Sentence-transformer embeddings
- Chunked topic assignment
- Topic-label export
- Saved BERTopic model
- Excel topic summary
- Interactive HTML chart

See the [BERTopic README](bertopic-chunked-pipeline/README.md) for setup and usage.

### Long-Text Emotion Classification

Location: [`emotion-classification/`](emotion-classification/)

Classifies English-language text records with DistilRoBERTa. Long records can be processed using overlapping token windows or a faster head-and-tail method.

Key capabilities:

- CSV and Excel input
- CPU-based execution
- File picker and command-line input
- Sliding-window classification
- Head-and-tail classification
- Per-label confidence scores
- Streaming CSV output
- Row-level error preservation

See the [emotion-classification README](emotion-classification/README.md) for setup and usage.

### LLM-Assisted Thematic Analysis Pipeline

Location: [`lata-llm-lms-pipeline/LATA_pipeline_clean/`](lata-llm-lms-pipeline/LATA_pipeline_clean/)

Runs a multi-stage thematic analysis using fixed research prompts and the OpenAI API.

Key capabilities:

- Batched CSV processing
- Source-record tracking
- Automatic retries and rate-limit backoff
- Checkpointing and resume support
- Rolling analysis memory
- Theme refinement and sub-theme generation
- Combined CSV export

The included prompts were developed for a specific research application. Preserve them when reproducing that analysis. Review and replace the research-specific prompt context when adapting the workflow to a different domain.

See the [LATA README](lata-llm-lms-pipeline/LATA_pipeline_clean/README.md) for setup and usage.

### Quote Verification Table

File: [`lata-llm-quote-verification.html`](lata-llm-quote-verification.html)

A standalone sortable HTML report for reviewing source quotations and their matching source text. Open it directly in a web browser; no server is required.

## Repository structure

```text
.
|-- bertopic-chunked-pipeline/
|   |-- .gitignore
|   |-- README.md
|   |-- bertopic_chunked.py
|   `-- requirements.txt
|-- emotion-classification/
|   |-- .gitignore
|   |-- README.md
|   |-- emotion_classification.py
|   `-- requirements.txt
|-- lata-llm-lms-pipeline/
|   `-- LATA_pipeline_clean/
|       |-- .env.example
|       |-- .gitignore
|       |-- README.md
|       |-- requirements.txt
|       `-- run_lata.py
|-- .gitignore
|-- LICENSE
|-- README.md
`-- lata-llm-quote-verification.html
```

## General setup

Python 3.10 or newer is recommended. Create a separate virtual environment for each workflow to avoid dependency conflicts.

Windows PowerShell example:

```powershell
cd path\to\selected-workflow
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

macOS or Linux example:

```bash
cd path/to/selected-workflow
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Hardware and connectivity

A GPU is not required for the included local workflows. Larger datasets may take substantially longer on CPU, particularly during embedding generation and sliding-window classification.

Internet access is required for initial model downloads. The LATA workflow also requires internet access during analysis because it uses the OpenAI API.

## Data handling

- The BERTopic and emotion-classification workflows process records locally after their models are downloaded.
- The LATA workflow sends prompt content to an external API.
- Review data-governance, privacy, and contractual requirements before processing sensitive or restricted text.
- Keep API keys and unpublished datasets out of version control.

## Reproducibility

For reproducible analysis, preserve:

- The exact input data and row order
- Python and dependency versions
- Model names and model revisions
- Script configuration and command-line arguments
- Prompt text used for API-based analysis
- Intermediate outputs and checkpoints

Model outputs may still vary across dependency versions, model revisions, hardware environments, or external API updates.

## Review requirements

Automated labels, topics, themes, sub-themes, confidence scores, and quote matches should be reviewed before they are treated as final analytical findings. Common review targets include misclassification, missing context, duplicate concepts, unsupported interpretations, and incorrect source attribution.

## License

This repository is released under the MIT License. See [`LICENSE`](LICENSE).
