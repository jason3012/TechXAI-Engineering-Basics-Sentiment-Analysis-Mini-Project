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

## Output

```
PREDICTION RESULTS
--------------------------------------------------
1. [PASS] Positive → Positive (0.988)  This product is absolutely amazing and exceeded all my expectations!
2. [PASS] Positive → Positive (0.988)  I had an amazing day because everything went perfectly.
3. [PASS] Positive → Positive (0.988)  The team delivered outstanding results and I couldn't be happier.
4. [PASS] Positive → Positive (0.469)  Just got into the program I applied for and I still can't believe it.
5. [PASS] Negative → Negative (0.957)  This is literally the worst service I have ever experienced in my life.
6. [PASS] Negative → Negative (0.892)  I hate doing all of this work in one day.
7. [PASS] Negative → Negative (0.950)  The food tasted horrible and way too expensive.
8. [PASS] Negative → Negative (0.950)  The client is insanely rude and I don't want to work with them anymore.
9. [PASS] Neutral → Neutral (0.948)  The meeting is scheduled for Tuesday at 3pm.
10. [PASS] Neutral → Neutral (0.455)  Lemonade is sweet and sour.
11. [PASS] Neutral → Neutral (0.856)  I need to talk to the professor today.
12. [PASS] Neutral → Neutral (0.626)  I bought oranges at the store.

12/12 correct


BASIC INPUT CHECKS
--------------------------------------------------
PASS (''): Input text must not be empty.
PASS (None): Expected str, got NoneType.
PASS (67): Expected str, got int.

INCORRECT/UNCERTAIN
--------------------------------------------------
[FAIL] "Oh great, another Monday."  (expected Negative, got Positive)
pos=0.586 neu=0.256 neg=0.158
[FAIL] "This could have been worse."  (expected Positive, got Negative)
pos=0.009 neu=0.119 neg=0.872
```

---

## Edge Cases

Two sentences the model struggled with:

**"Oh great, another Monday."** expected Negative, got Positive (0.586). The model picks up on "great" and misses the sarcasm. The low score at least shows it wasn't confident. Sarcasm is generally hard for these models without more context.

**"This could have been worse."** expected Positive, got Negative (0.872). "Worse" pulls it negative even though the sentence is implying things turned out okay, hence a positive expecation. This one is surprising given how high the confidence was.
