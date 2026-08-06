"""Classify emotions in CSV or Excel text records using DistilRoBERTa.

The script supports two strategies for records longer than the model context:

1. sliding_window: classify overlapping token windows and average scores.
2. head_tail: classify the beginning and end of the record together.

By default, both strategies are run on CPU and results are written to an
``emotion_classification_results`` directory next to the input file.
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
REQUIRED_COLUMNS = {"post_id", "title", "selftext"}
DEFAULT_STRIDE = 256

ScoreList = list[dict[str, float | str]]
ClassifierFunction = Callable[[str], ScoreList]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Classify emotions in CSV or Excel records with a CPU-based "
            "DistilRoBERTa model."
        )
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        type=Path,
        help="CSV or XLSX input file. A file picker opens when omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Output directory. Defaults to an emotion_classification_results "
            "folder beside the input file."
        ),
    )
    parser.add_argument(
        "--method",
        choices=("sliding_window", "head_tail", "both"),
        default="both",
        help="Long-text strategy to run. Default: both.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=DEFAULT_STRIDE,
        help="Sliding-window stride in tokens. Default: 256.",
    )
    parser.add_argument(
        "--model",
        default=MODEL_NAME,
        help=f"Hugging Face model name or local model path. Default: {MODEL_NAME}",
    )
    return parser.parse_args()


def choose_input_file() -> Path | None:
    """Open a file picker and return the selected path."""
    try:
        from tkinter import Tk, filedialog
    except ImportError as exc:
        raise RuntimeError(
            "Tkinter is unavailable. Pass the input path on the command line."
        ) from exc

    root = Tk()
    root.withdraw()
    try:
        selected = filedialog.askopenfilename(
            filetypes=[("CSV and Excel files", "*.csv *.xlsx")],
            title="Select a dataset",
        )
        return Path(selected) if selected else None
    finally:
        root.destroy()


def load_dataset(file_path: Path) -> pd.DataFrame:
    """Load and validate a CSV or XLSX dataset."""
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        dataframe = pd.read_csv(file_path)
    elif suffix == ".xlsx":
        dataframe = pd.read_excel(file_path)
    else:
        raise ValueError("Input must be a .csv or .xlsx file.")

    missing = sorted(REQUIRED_COLUMNS.difference(dataframe.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if dataframe.empty:
        raise ValueError("The input dataset contains no rows.")

    return dataframe


def build_classifier(model_name: str):
    """Load the tokenizer, model, and CPU text-classification pipeline."""
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    classifier = pipeline(
        task="text-classification",
        model=model,
        tokenizer=tokenizer,
        device=-1,
        top_k=None,
        truncation=True,
    )
    return tokenizer, classifier


def model_content_limit(tokenizer) -> int:
    """Return the usable number of content tokens after special tokens."""
    configured_limit = getattr(tokenizer, "model_max_length", 512)
    if configured_limit is None or configured_limit > 100_000:
        configured_limit = 512
    special_tokens = tokenizer.num_special_tokens_to_add(pair=False)
    return max(1, int(configured_limit) - special_tokens)


def normalize_scores(raw_result) -> ScoreList:
    """Normalize pipeline output to a list of label/score dictionaries."""
    if raw_result and isinstance(raw_result[0], list):
        raw_result = raw_result[0]
    return [
        {"label": str(item["label"]), "score": float(item["score"])}
        for item in raw_result
    ]


def classify_short_text(text: str, classifier) -> ScoreList:
    return normalize_scores(classifier(text))


def sliding_window_classify(
    text: str,
    tokenizer,
    classifier,
    max_content_tokens: int,
    stride: int,
) -> ScoreList:
    """Average emotion scores across overlapping token windows."""
    token_ids = tokenizer.encode(text, add_special_tokens=False, truncation=False)
    if len(token_ids) <= max_content_tokens:
        return classify_short_text(text, classifier)

    if stride <= 0:
        raise ValueError("Stride must be greater than zero.")
    if stride > max_content_tokens:
        raise ValueError(
            f"Stride ({stride}) cannot exceed the content-token limit "
            f"({max_content_tokens})."
        )

    totals: dict[str, float] = {}
    chunk_count = 0

    for start in range(0, len(token_ids), stride):
        window = token_ids[start : start + max_content_tokens]
        if not window:
            break

        chunk_text = tokenizer.decode(window, skip_special_tokens=True)
        for item in classify_short_text(chunk_text, classifier):
            label = str(item["label"])
            totals[label] = totals.get(label, 0.0) + float(item["score"])

        chunk_count += 1
        if start + max_content_tokens >= len(token_ids):
            break

    if chunk_count == 0:
        raise RuntimeError("No token windows were generated for the record.")

    return [
        {"label": label, "score": total / chunk_count}
        for label, total in totals.items()
    ]


def head_tail_classify(
    text: str,
    tokenizer,
    classifier,
    max_content_tokens: int,
) -> ScoreList:
    """Classify a token-balanced combination of the record head and tail."""
    token_ids = tokenizer.encode(text, add_special_tokens=False, truncation=False)
    if len(token_ids) <= max_content_tokens:
        return classify_short_text(text, classifier)

    head_size = max_content_tokens // 2
    tail_size = max_content_tokens - head_size
    combined = token_ids[:head_size] + token_ids[-tail_size:]
    combined_text = tokenizer.decode(combined, skip_special_tokens=True)
    return classify_short_text(combined_text, classifier)


def record_text(row: pd.Series) -> str:
    title = "" if pd.isna(row["title"]) else str(row["title"])
    selftext = "" if pd.isna(row["selftext"]) else str(row["selftext"])
    return f"{title} {selftext}".strip()


def ordered_labels(classifier) -> list[str]:
    scores = classify_short_text("test", classifier)
    return [str(item["label"]) for item in scores]


def export_results(
    dataframe: pd.DataFrame,
    output_path: Path,
    classify_func: ClassifierFunction,
    method_name: str,
    labels: Iterable[str],
) -> tuple[int, int]:
    """Classify every row and stream complete results to a CSV file."""
    label_list = list(labels)
    score_headers = [f"{label.lower()}_score" for label in label_list]
    fieldnames = list(dataframe.columns) + [
        "emotion",
        "score",
        "method",
        *score_headers,
        "processing_error",
    ]

    successful = 0
    failed = 0
    total = len(dataframe)

    with output_path.open("w", newline="", encoding="utf-8-sig") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()

        for position, (_, row) in enumerate(dataframe.iterrows(), start=1):
            output_row = row.to_dict()
            post_id = output_row.get("post_id", "N/A")

            try:
                text = record_text(row)
                if not text:
                    raise ValueError("Both title and selftext are empty.")

                scores = classify_func(text)
                score_map = {str(item["label"]): float(item["score"]) for item in scores}
                top = max(scores, key=lambda item: float(item["score"]))

                output_row.update(
                    {
                        "emotion": str(top["label"]),
                        "score": f"{float(top['score']):.6f}",
                        "method": method_name,
                        "processing_error": "",
                    }
                )
                for label, header in zip(label_list, score_headers):
                    output_row[header] = f"{score_map.get(label, 0.0):.6f}"

                successful += 1
                print(
                    f"[{method_name}] {position}/{total} - post_id: {post_id} - "
                    f"emotion: {top['label']} ({float(top['score']):.2%})"
                )
            except Exception as exc:  # Continue while preserving the failed row.
                failed += 1
                output_row.update(
                    {
                        "emotion": "",
                        "score": "",
                        "method": method_name,
                        "processing_error": str(exc),
                    }
                )
                for header in score_headers:
                    output_row[header] = ""
                print(
                    f"[{method_name}] {position}/{total} - post_id: {post_id} - "
                    f"error: {exc}",
                    file=sys.stderr,
                )

            writer.writerow(output_row)
            output_file.flush()

    return successful, failed


def process_file(args: argparse.Namespace) -> list[Path]:
    input_path = args.input_file or choose_input_file()
    if input_path is None:
        print("No file selected.")
        return []

    input_path = input_path.expanduser().resolve()
    if not input_path.is_file():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    dataframe = load_dataset(input_path)
    tokenizer, classifier = build_classifier(args.model)
    max_content_tokens = model_content_limit(tokenizer)

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else input_path.parent / "emotion_classification_results"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    labels = ordered_labels(classifier)
    methods = (
        ["sliding_window", "head_tail"]
        if args.method == "both"
        else [args.method]
    )

    written_files: list[Path] = []
    for method in methods:
        output_path = output_dir / f"emotion_results_{method}_{timestamp}.csv"
        if method == "sliding_window":
            classify_func = lambda text: sliding_window_classify(
                text,
                tokenizer,
                classifier,
                max_content_tokens,
                args.stride,
            )
        else:
            classify_func = lambda text: head_tail_classify(
                text,
                tokenizer,
                classifier,
                max_content_tokens,
            )

        print(f"Starting {method} classification for {len(dataframe):,} rows.")
        successful, failed = export_results(
            dataframe=dataframe,
            output_path=output_path,
            classify_func=classify_func,
            method_name=method,
            labels=labels,
        )
        print(
            f"Completed {method}: {successful:,} successful, {failed:,} failed."
        )
        print(f"Saved: {output_path}")
        written_files.append(output_path)

    return written_files


def main() -> int:
    args = parse_args()
    try:
        written_files = process_file(args)
    except KeyboardInterrupt:
        print("\nStopped by user.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    return 0 if written_files else 1


if __name__ == "__main__":
    raise SystemExit(main())
