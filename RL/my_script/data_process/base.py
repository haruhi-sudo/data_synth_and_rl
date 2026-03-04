"""Shared boilerplate for converting JSON datasets to Parquet format."""

import argparse
import json
import os
from typing import Callable, Optional

from datasets import Dataset


def convert_dataset(
    train_source: str,
    test_source: str,
    output_dir: str,
    make_map_fn: Callable,
    preprocess_fn: Optional[Callable] = None,
):
    """Common load -> preprocess -> map -> save pipeline.

    Args:
        train_source: Path to training data JSON file.
        test_source: Path to test/validation data JSON file.
        output_dir: Directory to write train.parquet and val.parquet.
        make_map_fn: Factory function that takes split name and returns a map function.
        preprocess_fn: Optional function to preprocess raw data list before Dataset creation.
                       Should accept and return a list of examples.
    """
    os.makedirs(output_dir, exist_ok=True)

    for split, source_path in [("train", train_source), ("test", test_source)]:
        with open(source_path, "r") as f:
            data = json.load(f)

        if preprocess_fn is not None:
            data = preprocess_fn(data)

        dataset = Dataset.from_list(data)
        dataset = dataset.map(function=make_map_fn(split), with_indices=True)

        filename = "train.parquet" if split == "train" else "val.parquet"
        dataset.to_parquet(os.path.join(output_dir, filename))


def make_base_parser(
    default_output_dir: str = "my_data/output",
    default_train_source: str = "",
    default_test_source: str = "",
) -> argparse.ArgumentParser:
    """Create a base argument parser with common arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default=default_output_dir)
    parser.add_argument("--train_data_source", default=default_train_source)
    parser.add_argument("--test_data_source", default=default_test_source)
    return parser
