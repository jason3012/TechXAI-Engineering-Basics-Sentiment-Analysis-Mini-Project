# Sentiment Analysis

Classifies text as **Positive**, **Negative**, or **Neutral** using [`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest), a RoBERTa model fine-tuned on ~124M tweets.

---

## Setup

```bash
pip3 install transformers torch
```

The model (~500 MB) downloads on first run and caches afterward. You'll probably see a warning about pooler weights — that's normal, doesn't affect anything.

---

## Usage

```bash
python3 sentiment_analyzer.py "I really enjoyed this!"
python3 test_sentiment.py   # runs the full test suite
```

---

## Results

12/12 correct on the main test set (4 positive, 4 negative, 4 neutral). Confidence scores mostly in the 0.90s for the clear-cut ones.

---

## Edge Cases

Two sentences the model struggled with:

**"Oh great, another Monday."** — expected Negative, got Positive (0.5855). The model picks up on "great" and misses the sarcasm. The low score at least shows it wasn't confident. Sarcasm is generally hard for these models without more context.

**"This could have been worse."** — expected Neutral, got Negative (0.8721). "Worse" pulls it negative even though the sentence is basically saying things are fine. This one surprised me a bit since the confidence was so high.
