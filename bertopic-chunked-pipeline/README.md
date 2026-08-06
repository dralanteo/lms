# Chunked BERTopic Pipeline

This workflow performs BERTopic analysis on a CSV or Parquet dataset. It fits a topic model on the first chunk of records, then uses that fitted model to assign topics to all remaining chunks.

The chunked design reduces peak processing demands compared with embedding and transforming the entire dataset in one operation. Model fitting still occurs only on the first chunk, so that chunk should be representative of the full dataset.

## Required input columns

The selected dataset must contain:

- `title`
- `selftext`

Duplicate records are removed using the combination of `title` and `selftext`. The fields are combined and cleaned before modeling. Records with no remaining text are removed.

## Installation

Python 3.10 or newer is recommended.

### Windows PowerShell

```powershell
python -m venv bertopic_env
.\bertopic_env\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS or Linux

```bash
python3 -m venv bertopic_env
source bertopic_env/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run with the file picker

```powershell
python bertopic_chunked.py
```

## Run from the command line

```powershell
python bertopic_chunked.py "path\to\dataset.csv"
```

Change the chunk size:

```powershell
python bertopic_chunked.py "path\to\dataset.parquet" --chunk-size 100000
```

View all options:

```powershell
python bertopic_chunked.py --help
```

## Outputs

Outputs are saved beside the selected input file:

- `<input_name>_bertopic_topic_labels.csv` or `.parquet`
- `<input_name>_bertopic_model/`
- `<input_name>_bertopic_topic_words.xlsx`
- `<input_name>_bertopic_barchart.html`

## Default model settings

- Embedding model: `all-distilroberta-v1`
- Chunk size: `200,000`
- HDBSCAN minimum cluster size: `20`
- HDBSCAN minimum samples: `10`
- UMAP neighbors: `15`
- UMAP components: `5`
- Random state: `0`

## Important behavior

The first chunk is used to fit the topic structure. Later chunks are transformed against that structure rather than used to create new topics. Input ordering and first-chunk composition can therefore affect the results.

The workflow can run without a GPU. Processing time depends heavily on dataset size, record length, CPU performance, and available memory.
