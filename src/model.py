from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def make_model() -> Pipeline:
	return Pipeline(
		[
			(
				"tfidf",
				TfidfVectorizer(
					lowercase=True,
					ngram_range=(1, 2), # Capture single words and two-word phrases
					min_df=2,   # Ignore terms appearing in fewer than 2 documents
					sublinear_tf=True,  # Apply log scaling to term frequency
)
				),
			),
			(
				"classifier",
				LogisticRegression(
					max_iter=500,   # Allow up to 500 optimization iterations for convergence
					random_state=42,
					solver="lbfgs",  # Use the L-BFGS optimization algorithm
				),
			),
		]
	)
