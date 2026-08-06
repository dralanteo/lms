"""Fit BERTopic on the first data chunk and transform remaining chunks."""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import nltk
import pandas as pd
import plotly.io as pio
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import CountVectorizer
from umap import UMAP

REQUIRED_COLUMNS = {"title", "selftext"}
DEFAULT_CHUNK_SIZE = 200_000
DEFAULT_EMBEDDING_MODEL = "all-distilroberta-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run chunked BERTopic analysis on a CSV or Parquet dataset."
    )
    parser.add_argument(
        "input_file",
        nargs="?",
        type=Path,
        help="CSV or Parquet input file. A file picker opens when omitted.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"Rows per chunk. Default: {DEFAULT_CHUNK_SIZE:,}.",
    )
    parser.add_argument(
        "--embedding-model",
        default=DEFAULT_EMBEDDING_MODEL,
        help=f"Sentence-transformer model. Default: {DEFAULT_EMBEDDING_MODEL}.",
    )
    return parser.parse_args()


def choose_input_file() -> Path | None:
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
            filetypes=[("CSV and Parquet files", "*.csv *.parquet")],
            title="Select a dataset",
        )
        return Path(selected) if selected else None
    finally:
        root.destroy()


def ensure_stopwords() -> list[str]:
    try:
        return stopwords.words("english")
    except LookupError:
        print("Downloading NLTK English stopwords...")
        nltk.download("stopwords", quiet=True)
        return stopwords.words("english")


def clean_text(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = re.sub(r"&amp;#x200b;|&amp;", " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\[.*?\]\((.*?)\)", r"\1", text)
    text = re.sub(r"[*>#`]", " ", text)
    text = re.sub(r"_x000d_|& x200b;", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_dataset(file_path: Path) -> tuple[pd.DataFrame, str]:
    suffix = file_path.suffix.lower()
    if suffix == ".csv":
        dataframe = pd.read_csv(file_path)
    elif suffix == ".parquet":
        dataframe = pd.read_parquet(file_path)
    else:
        raise ValueError("Input must be a .csv or .parquet file.")

    missing = sorted(REQUIRED_COLUMNS.difference(dataframe.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    dataframe = dataframe.drop_duplicates(
        subset=["title", "selftext"], keep="first"
    ).reset_index(drop=True)
    dataframe["text"] = (
        dataframe["title"].fillna("").astype(str)
        + ". "
        + dataframe["selftext"].fillna("").astype(str)
    ).map(clean_text)
    dataframe = dataframe[dataframe["text"].str.len() > 0].reset_index(drop=True)

    if dataframe.empty:
        raise ValueError("No non-empty text records remain after preprocessing.")

    return dataframe, suffix


def build_topic_model(english_stopwords: list[str]) -> BERTopic:
    return BERTopic(
        embedding_model=None,
        representation_model=KeyBERTInspired(
            top_n_words=10, nr_repr_docs=10, random_state=0
        ),
        ctfidf_model=ClassTfidfTransformer(reduce_frequent_words=True),
        vectorizer_model=CountVectorizer(stop_words=english_stopwords),
        hdbscan_model=HDBSCAN(
            min_cluster_size=20,
            metric="euclidean",
            min_samples=10,
            cluster_selection_method="eom",
            prediction_data=True,
        ),
        umap_model=UMAP(
            n_neighbors=15,
            n_components=5,
            min_dist=0.0,
            metric="cosine",
            random_state=0,
        ),
        verbose=True,
    )


def run_pipeline(input_path: Path, chunk_size: int, embedding_model_name: str) -> None:
    if chunk_size <= 0:
        raise ValueError("Chunk size must be greater than zero.")

    dataframe, suffix = load_dataset(input_path)
    total_rows = len(dataframe)
    total_chunks = (total_rows + chunk_size - 1) // chunk_size
    print(f"Total records after preprocessing: {total_rows:,}")
    print(f"Chunks: {total_chunks:,}")

    english_stopwords = ensure_stopwords()
    embedding_model = SentenceTransformer(embedding_model_name)
    topic_model = build_topic_model(english_stopwords)

    first_end = min(chunk_size, total_rows)
    first_texts = dataframe.loc[: first_end - 1, "text"].tolist()

    print("Fitting BERTopic on the first chunk...")
    started = time.time()
    first_embeddings = embedding_model.encode(first_texts, show_progress_bar=True)
    first_topics, _ = topic_model.fit_transform(first_texts, first_embeddings)
    all_topics = list(first_topics)
    print(f"Initial fit completed in {(time.time() - started) / 60:.2f} minutes.")

    for chunk_index in range(1, total_chunks):
        start_index = chunk_index * chunk_size
        end_index = min((chunk_index + 1) * chunk_size, total_rows)
        print(
            f"Processing chunk {chunk_index + 1}/{total_chunks} "
            f"(rows {start_index:,} through {end_index - 1:,})"
        )

        chunk_texts = dataframe.loc[start_index : end_index - 1, "text"].tolist()
        chunk_embeddings = embedding_model.encode(
            chunk_texts, show_progress_bar=True
        )
        chunk_topics, _ = topic_model.transform(chunk_texts, chunk_embeddings)
        all_topics.extend(list(chunk_topics))

    if len(all_topics) != len(dataframe):
        raise RuntimeError(
            "Topic assignment count does not match the processed row count."
        )

    dataframe["Assigned Topic"] = all_topics
    output_dir = input_path.parent
    base_name = input_path.stem

    labels_path = output_dir / f"{base_name}_bertopic_topic_labels{suffix}"
    if suffix == ".csv":
        dataframe.to_csv(labels_path, index=False)
    else:
        dataframe.to_parquet(labels_path, index=False)

    model_path = output_dir / f"{base_name}_bertopic_model"
    topic_model.save(str(model_path), serialization="pickle")

    topic_words_path = output_dir / f"{base_name}_bertopic_topic_words.xlsx"
    topic_model.get_topic_info().to_excel(topic_words_path, index=False)

    chart_path = output_dir / f"{base_name}_bertopic_barchart.html"
    pio.write_html(
        topic_model.visualize_barchart(),
        file=str(chart_path),
        auto_open=False,
    )

    print("Processing complete.")
    print(f"Model: {model_path}")
    print(f"Labels: {labels_path}")
    print(f"Topic words: {topic_words_path}")
    print(f"Bar chart: {chart_path}")


def main() -> int:
    args = parse_args()
    input_path = args.input_file or choose_input_file()
    if input_path is None:
        print("No file selected.")
        return 1

    input_path = input_path.expanduser().resolve()
    if not input_path.is_file():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        return 1

    try:
        run_pipeline(input_path, args.chunk_size, args.embedding_model)
    except KeyboardInterrupt:
        print("\nStopped by user.", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
