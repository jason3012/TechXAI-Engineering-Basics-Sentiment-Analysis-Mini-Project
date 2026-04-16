import sys
from transformers import pipeline


MODEL_ID = "cardiffnlp/twitter-roberta-base-sentiment-latest"


class SentimentAnalyzer:

    def __init__(self):
        self._pipe = pipeline(
            "sentiment-analysis",
            model=MODEL_ID,
            top_k=None,  # return all 3 label scores
            truncation=True,
            max_length=512,
        )

    def analyze(self, text):
        if not isinstance(text, str):
            raise TypeError(f"Expected str, got {type(text).__name__}.")
        if not text.strip():
            raise ValueError("Input text must not be empty.")

        raw = self._pipe(text)[0]
        scores = {item["label"].lower(): item["score"] for item in raw}
        top = max(raw, key=lambda x: x["score"])

        return {
            "sentiment": top["label"].capitalize(),
            "score": round(top["score"], 4),
            "positive": round(scores.get("positive", 0.0), 4),
            "neutral": round(scores.get("neutral", 0.0), 4),
            "negative": round(scores.get("negative", 0.0), 4),
        }


def main():
    analyzer = SentimentAnalyzer()

    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        text = input("Enter text to analyze: ")

    try:
        result = analyzer.analyze(text)
    except (TypeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"\n{result['sentiment']} ({result['score']:.3f})")
    print(f"pos={result['positive']:.3f}  neu={result['neutral']:.3f}  neg={result['negative']:.3f}")


if __name__ == "__main__":
    main()
