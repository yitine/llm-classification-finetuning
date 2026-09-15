import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data import TARGET_COLUMNS, load_test, load_train
from src.evaluate import multiclass_log_loss, validate_submission
from src.model import make_model

# Resolve this file's absolute path and go up two parent directories
ROOT = Path(__file__).resolve().parents[1]  


def run_baseline(
	train_path: Path,
	test_path: Path,
	sample_path: Path,
	submission_path: Path,
) -> float:
	texts, targets, train_dataframe = load_train(train_path)
	train_texts, validation_texts, train_targets, validation_targets = train_test_split(
		texts,
		targets,
		test_size=0.2,
		random_state=42,
		stratify=targets,  # Preserve the class distribution in the train and validation sets
	)

    # Check the class distribution of the full training dataset to ensure
    # the three target classes (model A, model B, and tie) are reasonably balanced.
	print("Class distribution:")
	print(train_dataframe[TARGET_COLUMNS].mean().rename(index={
		"winner_model_a": "model_a",
		"winner_model_b": "model_b",
		"winner_tie": "tie",
	}).to_string())

	model = make_model()
	model.fit(train_texts, train_targets)
	validation_probabilities = model.predict_proba(validation_texts)
	validation_loss = multiclass_log_loss(validation_targets, validation_probabilities)
	print(f"Validation log loss: {validation_loss:.6f}")

	test_texts, test_dataframe = load_test(test_path)
	test_probabilities = model.predict_proba(test_texts)
	submission = pd.DataFrame(test_probabilities, columns=TARGET_COLUMNS)
	submission.insert(0, "id", test_dataframe["id"].to_numpy())
	submission_path.parent.mkdir(parents=True, exist_ok=True)
	submission.to_csv(submission_path, index=False)
	validate_submission(submission_path, sample_path)
	print(f"Validated submission: {submission_path}")
	return validation_loss


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train the TF-IDF logistic regression baseline")
	parser.add_argument("--train-path", type=Path, default=ROOT / "data/raw/train.csv")
	parser.add_argument("--test-path", type=Path, default=ROOT / "data/raw/test.csv")
	parser.add_argument(
		"--sample-path", type=Path, default=ROOT / "data/raw/sample_submission.csv"
	)
	parser.add_argument(
		"--submission-path", type=Path, default=ROOT / "submissions/tfidf_logreg.csv"
	)
	return parser.parse_args()


if __name__ == "__main__":
	arguments = parse_args()
	run_baseline(
		arguments.train_path,
		arguments.test_path,
		arguments.sample_path,
		arguments.submission_path,
	)
