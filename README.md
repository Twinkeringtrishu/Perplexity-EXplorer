# Perplexity Explorer

An interactive command-line tool that lets you experiment with next-word prediction, probability distributions, and perplexity using a readily available causal language model.

## Features
* **Input a sentence or short phrase**, with optional **manual word editing** before prediction.
* Predicts the **top-k next-word candidates** (default `5`) together with their probabilities.
* Computes the **perplexity** of the resulting sentence.
* Tracks a running **Score** (instead of a penalty label):
  * Correct top-1 prediction: **0** (no change)
  * Wrong top-1 prediction: **-1**
  * Answer *"I don't know"* / *"I'm not sure"* / *"idk"*: **+1**
* **Persistent score logging** to `score_log.json` (resumes your last score automatically unless `--reset-score` is passed).
* **Automated sentence generation** from the current sentence, with an option to feed the generated sentence straight into the next round.
* **Configurable model & candidate count** via CLI flags.

## Installation
```bash
# Create a virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Usage
```bash
python explorer.py
```

### Options
```bash
# Use a different Hugging Face causal LM (default: gpt2)
python explorer.py --model gpt2-medium

# Show top 10 next-word candidates
python explorer.py --top-k 10

# Log rounds to a custom file and start from score 0
python explorer.py --log-file my_scores.json --reset-score
```

## License
MIT – feel free to adapt and extend.
