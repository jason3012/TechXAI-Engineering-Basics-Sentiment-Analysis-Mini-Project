# Sentiment Analysis

Classifies text as **Positive**, **Negative**, or **Neutral** using [`cardiffnlp/twitter-roberta-base-sentiment-latest`](https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest), a RoBERTa model fine-tuned on ~124M tweets.

---

## Setup

```bash
pip3 install transformers torch
```

The model downloads on first run and caches afterward.

---

## Usage

```bash
python3 sentiment_analyzer.py "I really enjoyed this!"
python3 test_sentiment.py   # runs the full test of preset sentences
```

---

## Results

12/12 correct on the main test set (4 positive, 4 negative, 4 neutral). Confidence scores mostly in the 0.90s for the clear-cut ones.

---

## Edge Cases

Two sentences the model struggled with:

**"Oh great, another Monday."** expected Negative, got Positive (0.586). The model picks up on "great" and misses the sarcasm. The low score at least shows it wasn't confident. Sarcasm is generally hard for these models without more context.

**"This could have been worse."** expected Positive, got Negative (0.872). "Worse" pulls it negative even though the sentence is implying things turned out okay, hence a positive expecation. This one is surprising given how high the confidence was.
