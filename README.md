# Perplexity Explorer

An interactive command-line tool that lets you experiment with next-word prediction, probability distributions, and perplexity **with zero model downloads required**.

## Features
* **Zero-download built-in n-gram model (`ngram`)**: Starts instantly using an interpolated trigram/bigram/unigram model with add-*k* smoothing, and learns dynamically from every sentence or short phrase you enter.
* **Input a sentence or short phrase**, with optional **manual word editing** before prediction.
* Predicts the **top-k next-word candidates** (default `5`) together with their probabilities.
* Computes the **perplexity** of the resulting sentence.
* Tracks a running **Score**:
  * Correct top-1 prediction: **0** (no change)
  * Wrong top-1 prediction: **-1**
  * Answer *"I don't know"* / *"I'm not sure"* / *"idk"*: **+1**
* **Persistent score logging** to `score_log.json` (resumes your last score automatically unless `--reset-score` is passed).
* **Automated sentence/phrase generation**, with an option to feed the generated text straight into the next round.

## Usage
No extra installation is needed — run directly with Python 3:

```bash
python3 explorer.py
```

### Options
```bash
# Show top 10 next-word candidates
python3 explorer.py --top-k 10

# Train the built-in model on an extra plain-text file
python3 explorer.py --corpus my_sentences.txt

# Log rounds to a custom file and start from score 0
python3 explorer.py --log-file my_scores.json --reset-score
```

## License
MIT – feel free to adapt and extend.
