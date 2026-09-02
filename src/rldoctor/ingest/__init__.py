"""Log ingestion: framework logs in, canonical :class:`~rldoctor.schema.Run` out."""

from .base import load_csv, load_json, load_jsonl, load_run, run_from_records

__all__ = ["load_csv", "load_json", "load_jsonl", "load_run", "run_from_records"]
