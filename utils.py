import json
import math
import random
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

UNSURE_ANSWERS = {"i don't know", "i am not sure", "i'm not sure", "idk"}

DEFAULT_CORPUS = [
    "the quick brown fox jumps over the lazy dog",
    "the sun rises in the east and sets in the west",
    "a journey of a thousand miles begins with a single step",
    "she opened the door and walked into the quiet room",
    "he drank a cup of hot coffee in the morning",
    "the children played in the park until sunset",
    "we are learning how language models predict the next word",
    "every sentence has a probability and a perplexity score",
    "the rain fell softly on the green leaves",
    "they decided to take a short walk along the river",
    "good ideas often come from asking simple questions",
    "the cat sat on the warm mat near the window",
    "birds sing early in the morning before the city wakes",
    "reading books helps you discover new worlds and ideas",
    "the train arrived at the station right on time",
    "i am not sure what the next word will be",
    "short phrases are easy to test and explore",
    "time flies when you are working on something fun",
    "the sky turned bright orange as the sun went down",
    "practice makes progress every single day",
    "she wrote a quick note on the kitchen table",
    "the wind blew across the open field",
    "he smiled and said thank you for your help",
    "music can change your mood in just a few seconds",
    "always check your work before you share it",
]


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


class NGramLanguageModel:
    """Zero-download interpolated trigram/bigram/unigram language model with add-k smoothing."""

    def __init__(self, corpus: list[str] | None = None, smoothing: float = 0.5):
        self.smoothing = smoothing
        self.unigrams: Counter[str] = Counter()
        self.bigrams: dict[str, Counter[str]] = defaultdict(Counter)
        self.trigrams: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
        self.total_tokens = 0
        for line in corpus or DEFAULT_CORPUS:
            self.train_sentence(line)

    def train_sentence(self, text: str) -> None:
        """Update n-gram counts from a sentence or phrase."""
        tokens = _tokenize(text)
        if not tokens:
            return
        padded = ["<s>", "<s>", *tokens, "</s>"]
        for i in range(2, len(padded)):
            w2, w1, w = padded[i - 2], padded[i - 1], padded[i]
            self.unigrams[w] += 1
            self.bigrams[w1][w] += 1
            self.trigrams[(w2, w1)][w] += 1
            self.total_tokens += 1

    def _prob_next(self, w2: str, w1: str, word: str) -> float:
        vocab_size = max(len(self.unigrams), 1)
        k = self.smoothing

        # Unigram probability with add-k smoothing
        p_uni = (self.unigrams[word] + k) / (self.total_tokens + k * vocab_size)

        # Bigram probability
        bi_counts = self.bigrams.get(w1)
        if bi_counts:
            p_bi = (bi_counts[word] + k) / (sum(bi_counts.values()) + k * vocab_size)
        else:
            p_bi = p_uni

        # Trigram probability
        tri_counts = self.trigrams.get((w2, w1))
        if tri_counts:
            p_tri = (tri_counts[word] + k) / (sum(tri_counts.values()) + k * vocab_size)
        else:
            p_tri = p_bi

        return 0.6 * p_tri + 0.3 * p_bi + 0.1 * p_uni

    def predict_next(self, text: str, top_k: int = 5) -> tuple[list[str], list[float]]:
        """Return the top-k next-word candidates and their probabilities."""
        tokens = _tokenize(text)
        w2 = tokens[-2] if len(tokens) >= 2 else "<s>"
        w1 = tokens[-1] if len(tokens) >= 1 else "<s>"

        vocab = [w for w in self.unigrams if w != "</s>"]
        if not vocab:
            return ["the"], [1.0]

        scored = [(w, self._prob_next(w2, w1, w)) for w in vocab]
        total = sum(p for _, p in scored) or 1.0
        scored = [(w, p / total) for w, p in scored]
        scored.sort(key=lambda item: item[1], reverse=True)
        top = scored[:top_k]
        return [w for w, _ in top], [p for _, p in top]

    def compute_perplexity(self, text: str) -> float:
        """Compute sentence perplexity from average negative log-likelihood."""
        tokens = _tokenize(text)
        if not tokens:
            return float("inf")
        padded = ["<s>", "<s>", *tokens]
        log_prob_sum = 0.0
        for i in range(2, len(padded)):
            p = max(self._prob_next(padded[i - 2], padded[i - 1], padded[i]), 1e-12)
            log_prob_sum += math.log(p)
        avg_nll = -log_prob_sum / len(tokens)
        return math.exp(avg_nll)

    def generate_sentence(self, prompt: str = "", max_new_words: int = 10) -> str:
        """Sample a continuation or new sentence/phrase from the n-gram distribution."""
        tokens = _tokenize(prompt)
        generated = list(tokens)
        vocab = list(self.unigrams.keys())
        if not vocab:
            return prompt

        for _ in range(max_new_words):
            w2 = generated[-2] if len(generated) >= 2 else "<s>"
            w1 = generated[-1] if len(generated) >= 1 else "<s>"
            weights = [self._prob_next(w2, w1, w) for w in vocab]
            next_word = random.choices(vocab, weights=weights, k=1)[0]
            if next_word == "</s>":
                if len(generated) > len(tokens):
                    break
                continue
            generated.append(next_word)
        return " ".join(generated)


def load_model(model_name: str = "ngram", corpus_path: str | None = None):
    """Load the built-in zero-download n-gram model (default) or a HuggingFace model."""
    if model_name == "ngram":
        corpus = list(DEFAULT_CORPUS)
        if corpus_path:
            lines = Path(corpus_path).read_text(encoding="utf-8").splitlines()
            corpus.extend(line.strip() for line in lines if line.strip())
        return NGramLanguageModel(corpus=corpus), None

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    model.to(device)
    model.eval()
    return model, tokenizer


def predict_next(model, tokenizer, text: str, top_k: int = 5):
    """Return the top-k next-word candidates and their probabilities."""
    if isinstance(model, NGramLanguageModel):
        return model.predict_next(text, top_k=top_k)

    import torch

    device = next(model.parameters()).device
    inputs = tokenizer.encode(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(inputs)
        logits = outputs.logits
    next_token_logits = logits[0, -1, :]
    probs = torch.softmax(next_token_logits, dim=-1)
    top_probs, top_indices = torch.topk(probs, top_k)
    candidates = [
        tokenizer.decode([idx], clean_up_tokenization_spaces=True).strip()
        for idx in top_indices.tolist()
    ]
    return candidates, top_probs.tolist()


def compute_perplexity(model, tokenizer, text: str) -> float:
    """Compute perplexity of a given string."""
    if isinstance(model, NGramLanguageModel):
        return model.compute_perplexity(text)

    import torch

    device = next(model.parameters()).device
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss.item()
    return torch.exp(torch.tensor(loss)).item()


def generate_sentence(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 12,
    temperature: float = 0.8,
    top_p: float = 0.9,
) -> str:
    """Sample a continuation or new sentence from the model."""
    if isinstance(model, NGramLanguageModel):
        return model.generate_sentence(prompt, max_new_words=max_new_tokens)

    import torch

    device = next(model.parameters()).device
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=top_p,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


def update_score(score: int, answer: str, top1: str) -> tuple[int, int, str]:
    """Update score based on the user's answer vs the model's top-1 guess.

    Returns (new_score, delta, status_message):
      - Unsure answer ('i don't know' / 'i am not sure' / 'idk'): +1
      - Correct top-1 guess: 0
      - Wrong top-1 guess: -1
    """
    normalized = answer.strip().lower()
    if normalized in UNSURE_ANSWERS:
        return score + 1, 1, "+1 added (unsure answer)."
    if normalized == top1.strip().lower():
        return score, 0, "Correct! No change to score."
    return score - 1, -1, "Wrong guess. -1 from score."


def ai_guess_next(
    model,
    tokenizer,
    prefix: str,
    top_k: int = 5,
    unsure_threshold: float = 0.015,
) -> tuple[str, bool, list[str], list[float]]:
    """Have the AI model guess the next word, or say 'I don't know' / 'I am not sure'
    when its top-1 probability is below unsure_threshold or the context is unseen.

    Returns (ai_guess, is_unsure, candidates, probs).
    """
    candidates, probs = predict_next(model, tokenizer, prefix, top_k=top_k)
    if not candidates or not probs:
        return "I don't know", True, [], []

    if isinstance(model, NGramLanguageModel):
        tokens = _tokenize(prefix)
        w2 = tokens[-2] if len(tokens) >= 2 else "<s>"
        w1 = tokens[-1] if len(tokens) >= 1 else "<s>"
        if w1 not in model.bigrams and (w2, w1) not in model.trigrams:
            return "I don't know", True, candidates, probs

    top1, top1_prob = candidates[0], probs[0]
    if top1_prob < unsure_threshold:
        phrase = "I am not sure" if top1_prob >= unsure_threshold * 0.6 else "I don't know"
        return phrase, True, candidates, probs

    return top1, False, candidates, probs


def score_ai_guess(score: int, ai_guess: str, actual_word: str) -> tuple[int, int, str]:
    """Score the AI's guess against the actual next word.

    - AI says 'I don't know' / 'I am not sure': +1
    - AI guesses right: 0
    - AI guesses wrong: -1
    """
    if ai_guess.strip().lower() in UNSURE_ANSWERS:
        return score + 1, 1, f"AI said '{ai_guess}' (+1 to score)."
    if ai_guess.strip().lower() == actual_word.strip().lower():
        return score, 0, f"AI guessed '{ai_guess}' right! (0 change)."
    return score - 1, -1, f"AI guessed '{ai_guess}' instead of '{actual_word}' (-1 from score)."


def load_score_log(log_path: str | Path) -> list[dict]:
    """Load existing score log entries from a JSON file, or return an empty list."""
    path = Path(log_path)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def append_score_log(log_path: str | Path, entry: dict) -> list[dict]:
    """Append a round entry with an ISO timestamp to the JSON score log."""
    path = Path(log_path)
    entries = load_score_log(path)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **entry,
    }
    entries.append(record)
    path.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    return entries
