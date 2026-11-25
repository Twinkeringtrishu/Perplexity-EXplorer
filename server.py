import argparse
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from utils import (
    ai_guess_next,
    append_score_log,
    load_model,
    load_score_log,
    score_ai_guess,
)

BASE_DIR = Path(__file__).resolve().parent


def _split_prefix_and_target(text: str) -> tuple[str, str] | None:
    tokens = re.findall(r"[a-zA-Z0-9']+", text.strip())
    if len(tokens) < 2:
        return None
    return " ".join(tokens[:-1]), tokens[-1].lower()


def make_handler(model, log_path: Path):
    score_state = {"score": 0}
    existing = load_score_log(log_path)
    if existing:
        score_state["score"] = existing[-1].get("score", 0)

    class PerplexityHandler(BaseHTTPRequestHandler):
        def _send_json(self, payload: dict, status: int = 200):
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> dict:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b"{}"
            return json.loads(raw.decode("utf-8"))

        def do_GET(self):
            if self.path in {"/", "/index.html"}:
                html = (BASE_DIR / "index.html").read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(html)))
                self.end_headers()
                self.wfile.write(html)
                return
            if self.path == "/api/state":
                history = load_score_log(log_path)
                score_state["score"] = history[-1].get("score", 0) if history else 0
                self._send_json({"score": score_state["score"], "history": history[-20:][::-1]})
                return
            self._send_json({"error": "Not found"}, status=404)

        def do_POST(self):
            data = self._read_json()

            if self.path == "/api/guess":
                text = str(data.get("text", "")).strip()
                top_k = max(1, min(int(data.get("top_k", 5)), 15))
                threshold = float(data.get("unsure_threshold", 0.015))
                split = _split_prefix_and_target(text)
                if not split:
                    self._send_json(
                        {"error": "Please enter at least 2 words so AI can guess the last word."},
                        status=400,
                    )
                    return

                prefix, actual_word = split
                ai_guess, is_unsure, candidates, probs = ai_guess_next(
                    model, None, prefix, top_k=top_k, unsure_threshold=threshold
                )
                new_score, delta, message = score_ai_guess(
                    score_state["score"], ai_guess, actual_word
                )
                score_state["score"] = new_score

                full_sentence = f"{prefix} {actual_word}"
                ppl = model.compute_perplexity(full_sentence)
                model.train_sentence(full_sentence)

                history = append_score_log(
                    log_path,
                    {
                        "model": "ngram",
                        "input": prefix,
                        "top_candidates": [
                            {"word": w, "probability": round(p, 6)}
                            for w, p in zip(candidates, probs)
                        ],
                        "ai_guess": ai_guess,
                        "actual_word": actual_word,
                        "score_change": delta,
                        "score": new_score,
                        "perplexity": round(ppl, 2),
                    },
                )
                self._send_json(
                    {
                        "prefix": prefix,
                        "actual_word": actual_word,
                        "ai_guess": ai_guess,
                        "is_unsure": is_unsure,
                        "candidates": [
                            {"word": w, "probability": round(p, 6)}
                            for w, p in zip(candidates, probs)
                        ],
                        "perplexity": round(ppl, 2),
                        "score": new_score,
                        "delta": delta,
                        "message": message,
                        "history": history[-20:][::-1],
                    }
                )
                return

            if self.path == "/api/perplexity":
                text = str(data.get("text", "")).strip()
                if not text:
                    self._send_json({"perplexity": None})
                    return
                ppl = model.compute_perplexity(text)
                self._send_json({"perplexity": round(ppl, 2)})
                return

            if self.path == "/api/generate":
                prompt = str(data.get("text", "")).strip()
                mode = str(data.get("mode", "sentence"))
                max_words = 4 if mode == "phrase" else 9
                generated = model.generate_sentence(prompt, max_new_words=max_words)
                ppl = model.compute_perplexity(generated)
                self._send_json({"generated": generated, "perplexity": round(ppl, 2)})
                return

            if self.path == "/api/reset":
                score_state["score"] = 0
                log_path.write_text("[]", encoding="utf-8")
                self._send_json({"score": 0, "history": []})
                return

            self._send_json({"error": "Not found"}, status=404)

        def log_message(self, format, *args):
            return

    return PerplexityHandler


def main():
    parser = argparse.ArgumentParser(description="Perplexity Explorer Web Server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8765, help="Port to listen on (default: 8765)")
    parser.add_argument("--log-file", default="score_log.json", help="Persistent score log file")
    parser.add_argument("--corpus", default=None, help="Optional extra plain-text corpus file")
    args = parser.parse_args()

    model, _ = load_model("ngram", corpus_path=args.corpus)
    handler = make_handler(model, BASE_DIR / args.log_file)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"Perplexity Explorer web app running at http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
