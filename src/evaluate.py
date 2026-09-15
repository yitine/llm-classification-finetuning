from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from src.data import TARGET_COLUMNS


def multiclass_log_loss(targets: pd.Series, probabilities: np.ndarray) -> float:
	return log_loss(targets, probabilities, labels=list(range(len(TARGET_COLUMNS))))


def validate_submission(
	submission_path: Union[str, Path], sample_path: Union[str, Path]
) -> None:
	submission = pd.read_csv(submission_path, dtype={"id": str})
	sample = pd.read_csv(sample_path, dtype={"id": str})
	expected_columns = ["id", *TARGET_COLUMNS]
	if submission.columns.tolist() != expected_columns:
		raise ValueError(f"Unexpected submission columns: {submission.columns.tolist()}")
	if submission.shape != sample.shape:
		raise ValueError("Submission shape does not match sample submission")
	if not submission["id"].equals(sample["id"]):
		raise ValueError("Submission IDs do not match sample submission IDs")
	probabilities = submission[TARGET_COLUMNS].to_numpy()
	if not np.isfinite(probabilities).all():
		raise ValueError("Submission contains non-finite probabilities")
	if (probabilities < 0).any() or (probabilities > 1).any():
		raise ValueError("Submission probabilities must be between 0 and 1")
	if not np.allclose(probabilities.sum(axis=1), 1.0):
		raise ValueError("Submission probabilities must sum to 1")
