"""
Quick test suite for SentimentAnalyzer.

Run:
    python3 test_sentiment.py
"""

from sentiment_analyzer import SentimentAnalyzer


TEST_SENTENCES = [
    # positive
    ("This product is absolutely amazing and exceeded all my expectations!", "Positive"),
    ("I had an amazing day because everything went perfectly.", "Positive"),
    ("The team delivered outstanding results and I couldn't be happier.", "Positive"),
    ("Just got into the program I applied for and I still can't believe it.", "Positive"),
    # negative
    ("This is literally the worst service I have ever experienced in my life.", "Negative"),
    ("I hate doing all of this work in one day.", "Negative"),
    ("The food tasted horrible and way too expensive.", "Negative"),
    ("The client is insanely rude and I don't want to work with them anymore.", "Negative"),
    # neutral
    ("The meeting is scheduled for Tuesday at 3pm.", "Neutral"),
    ("Lemonade is sweet and sour.", "Neutral"),
    ("I need to talk to the professor today.", "Neutral"),
    ("I bought oranges at the store.", "Neutral"),
]


def main() -> None:
    analyzer = SentimentAnalyzer()

    print("\nPREDICTION RESULTS")
    print("-" * 50)

    correct = 0
    for i, (text, expected) in enumerate(TEST_SENTENCES, 1):
        result = analyzer.analyze(text)
        predicted = result["sentiment"]
        status = "PASS" if predicted == expected else "FAIL"
        if status == "PASS":
            correct += 1
        print(f"{i}. [{status}] {expected} → {predicted} ({result['score']:.3f})  {text}")

    print(f"\n{correct}/{len(TEST_SENTENCES)} correct\n")

    print("\nBASIC INPUT CHECKS")
    print("-" * 50)

    # rejects bad input
    for bad in ["", None, 67]:
        try:
            analyzer.analyze(bad)
            print(f"FAIL: no exception for {bad!r}")
        except (TypeError, ValueError) as exc:
            print(f"PASS ({bad!r}): {exc}")

    print("\nINCORRECT/UNCERTAIN")
    print("-" * 50)

    # sarcastic, ambiguous
    for text, expected in [("Oh great, another Monday.", "Negative"),
                            ("This could have been worse.", "Positive")]:
        result = analyzer.analyze(text)
        predicted = result["sentiment"]
        match = "PASS" if predicted == expected else "FAIL"
        print(f"[{match}] \"{text}\"  (expected {expected}, got {predicted})")
        print(f"pos={result['positive']:.3f} neu={result['neutral']:.3f} neg={result['negative']:.3f}")


if __name__ == "__main__":
    main()
