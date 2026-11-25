import json
from datetime import datetime, timezone
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

UNSURE_ANSWERS = {"i don't know", "i am not sure", "i'm not sure", "idk"}


def load_model(model_name: str = "gpt2"):
    """Load a causal language model and its tokenizer.
    Returns (model, tokenizer) in eval mode on GPU if available, else CPU."""
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
    """Return the top-k next-token candidates and their probabilities for a text prefix."""
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
    """Compute perplexity of a given string from average negative log-likelihood."""
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
    max_new_tokens: int = 15,
    temperature: float = 0.8,
    top_p: float = 0.9,
) -> str:
    """Sample a continuation from the model starting from prompt."""
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
