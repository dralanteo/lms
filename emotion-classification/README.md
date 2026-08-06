# Emotion Classification for Long Text Records

This workflow assigns emotion labels and confidence scores to CSV or Excel records using the `j-hartmann/emotion-english-distilroberta-base` model from Hugging Face.

It runs on CPU by default and supports records longer than the model's context window through two processing methods:

- `sliding_window`: splits long text into overlapping token windows and averages the emotion scores.
- `head_tail`: combines the beginning and end of long text into one model-length input.

The default run executes both methods and creates a separate output file for each one.

## Required input columns

The input file must be a `.csv` or `.xlsx` file containing:

- `post_id`: a unique or source identifier retained in the output
- `title`: title text; blank values are allowed
- `selftext`: body text; blank values are allowed

Rows with both `title` and `selftext` empty are retained in the output and marked in the `processing_error` column.

## Installation

Python 3.10 or newer is recommended.

### Windows PowerShell

```powershell
python -m venv emotion_env
.\emotion_env\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS or Linux

```bash
python3 -m venv emotion_env
source emotion_env/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The model is downloaded automatically during the first run and then reused from the local Hugging Face cache.

## Run with the file picker

```powershell
python emotion_classification.py
```

Select a CSV or Excel file when prompted.

## Run from the command line

```powershell
python emotion_classification.py "path\to\dataset.csv"
```

Choose one processing method:

```powershell
python emotion_classification.py "path\to\dataset.csv" --method sliding_window
```

Specify an output directory:

```powershell
python emotion_classification.py "path\to\dataset.csv" --output-dir "path\to\results"
```

View all options:

```powershell
python emotion_classification.py --help
```

## Outputs

By default, files are written to an `emotion_classification_results` folder beside the input file:

```text
emotion_results_sliding_window_YYYYMMDD_HHMMSS.csv
emotion_results_head_tail_YYYYMMDD_HHMMSS.csv
```

Each output contains all original columns plus:

- `emotion`: highest-scoring emotion label
- `score`: confidence score for the selected label
- `method`: processing method used
- one score column for every emotion label
- `processing_error`: row-level error details when classification fails

The output file is written one row at a time and flushed after each record. This preserves completed work if a later row fails or the process is interrupted.

## Method selection

Use `sliding_window` when information throughout the full record may matter. It is more comprehensive but slower because long records require multiple model calls.

Use `head_tail` when speed matters and the beginning and end are likely to contain the most useful context. It requires one model call per record but may miss information in the middle.

Running `both` is useful for comparing the two approaches, but it approximately doubles processing time.

## Notes and limitations

- The included model is trained for English emotion classification.
- Model scores are predictions, not verified ground truth.
- Averaging window scores gives each window equal weight.
- Overlapping windows repeat some text by design.
- Review classifications before using them for high-stakes decisions or formal conclusions.
