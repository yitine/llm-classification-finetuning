import json
from pathlib import Path
from typing import Tuple, Union

import pandas as pd


TARGET_COLUMNS = ["winner_model_a", "winner_model_b", "winner_tie"]


def parse_conversation(value: str) -> str:
	"""Convert a JSON-encoded conversation into plain text."""
	conversation = json.loads(value)
	if not isinstance(conversation, list):
		raise ValueError("Conversation value must be a JSON list")
	return "\n".join(
		item if isinstance(item, str) else json.dumps(item, sort_keys=True)
		for item in conversation
	)


def build_text(dataframe: pd.DataFrame) -> pd.Series:
	"""Build one text document from each prompt and response pair."""
    # Use "unknown" if the model column is missing from the dataframe
	model_a = dataframe.get("model_a", pd.Series("unknown", index=dataframe.index))
	model_b = dataframe.get("model_b", pd.Series("unknown", index=dataframe.index))
	return (
		"prompt: "
		+ dataframe["prompt"].map(parse_conversation)
		+ "\nresponse_a: "
		+ dataframe["response_a"].map(parse_conversation)
		+ "\nresponse_b: "
		+ dataframe["response_b"].map(parse_conversation)
		+ "\nmodel_a: "
		+ model_a.fillna("unknown")   # Replace missing values within the model columns
		+ "\nmodel_b: "
		+ model_b.fillna("unknown")
	)


def load_train(path: Union[str, Path]) -> Tuple[pd.Series, pd.Series, pd.DataFrame]:
	dataframe = pd.read_csv(path, dtype={"id": str})
	texts = build_text(dataframe)
    # Convert one-hot labels to class indices: 0=A, 1=B, 2=Tie
	labels = dataframe[TARGET_COLUMNS].to_numpy().argmax(axis=1)
	return texts, pd.Series(labels, index=dataframe.index, name="target"), dataframe


def load_test(path: Union[str, Path]) -> Tuple[pd.Series, pd.DataFrame]:
	dataframe = pd.read_csv(path, dtype={"id": str})
	return build_text(dataframe), dataframe
