# Perplexity Explorer

An interactive **web application** (and CLI) that lets you experiment with next-word prediction, probability distributions, and perplexity **with zero model downloads required**.

## Features
* **Interactive Web UI** (`server.py` + `index.html`) — no external packages required.
* **Zero-download built-in n-gram model (`ngram`)**: Interpolated trigram/bigram/unigram model with add-*k* smoothing that learns dynamically from every sentence or short phrase you enter.
* **Manual word editing**: Click any word chip in the UI to edit words directly before predicting.
* **Top-k next-word probabilities & Perplexity** visualization.
* **Score tracking**:
  * Correct top-1 prediction: **0** (no change)
  * Wrong top-1 prediction: **-1**
  * *"I don't know"* / *"I am not sure"*: **+1**
* **Automated sentence & short phrase generation** with one-click reuse.
* **Persistent score logging** to `score_log.json`.

## Usage

### Web App (Recommended)
```bash
python3 server.py
```
Then open **http://127.0.0.1:8765** in your browser.

### CLI Mode
```bash
python3 explorer.py
```

## License
MIT – feel free to adapt and extend.
