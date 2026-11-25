import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from utils import (
    UNSURE_ANSWERS,
    append_score_log,
    load_model,
    load_score_log,
    update_score,
)

BASE_DIR = Path(__file__).resolve().parent


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

            if self.path == "/api/predict":
                text = str(data.get("text", "")).strip()
                top_k = max(1, min(int(data.get("top_k", 5)), 15))
                if not text:
                    self._send_json({"error": "Please enter a sentence or short phrase."}, status=400)
                    return
                candidates, probs = model.predict_next(text, top_k=top_k)
                ppl = model.compute_perplexity(text)
                self._send_json(
                    {
                        "text": text,
                        "candidates": [
                            {"word": w, "probability": round(p, 6)}
                            for w, p in zip(candidates, probs)
                        ],
                        "perplexity": round(ppl, 2),
                    }
                )
                return

            if self.path == "/api/answer":
                text = str(data.get("text", "")).strip()
                answer = str(data.get("answer", "")).strip()
                top_k = max(1, min(int(data.get("top_k", 5)), 15))
                if not text or not answer:
                    self._send_json({"error": "Both sentence and answer are required."}, status=400)
                    return

                candidates, probs = model.predict_next(text, top_k=top_k)
                top1 = candidates[0] if candidates else ""
                new_score, delta, message = update_score(score_state["score"], answer, top1)
                score_state["score"] = new_score

                if answer.lower() in UNSURE_ANSWERS:
                    full_sentence = text
                else:
                    full_sentence = f"{text} {answer}"

                ppl = model.compute_perplexity(full_sentence)
                model.train_sentence(full_sentence)

                history = append_score_log(
                    log_path,
                    {
                        "model": "ngram",
                        "input": text,
                        "top_candidates": [
                            {"word": w, "probability": round(p, 6)}
                            for w, p in zip(candidates, probs)
                        ],
                        "top1_prediction": top1,
                        "user_answer": answer,
                        "score_change": delta,
                        "score": new_score,
                        "perplexity": round(ppl, 2),
                    },
                )
                self._send_json(
                    {
                        "score": new_score,
                        "delta": delta,
                        "message": message,
                        "top1": top1,
                        "full_sentence": full_sentence,
                        "perplexity": round(ppl, 2),
                        "history": history[-20:][::-1],
                    }
                )
                return

            if self.path == "/api/generate":
                prompt = str(data.get("text", "")).strip()
                mode = str(data.get("mode", "sentence"))
                max_words = 5 if mode == "phrase" else 10
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
