import argparse
from utils import (
    NGramLanguageModel,
    UNSURE_ANSWERS,
    append_score_log,
    compute_perplexity,
    generate_sentence,
    load_model,
    load_score_log,
    predict_next,
    update_score,
)


def main():
    parser = argparse.ArgumentParser(description="Perplexity Explorer Interactive CLI")
    parser.add_argument(
        "--model",
        default="ngram",
        help="Model to use: 'ngram' (built-in, zero download) or a HuggingFace model ID",
    )
    parser.add_argument(
        "--corpus",
        default=None,
        help="Optional path to a plain-text corpus file to train the built-in n-gram model",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top next-word candidates to display (default: 5)",
    )
    parser.add_argument(
        "--log-file",
        default="score_log.json",
        help="Path to persistent JSON score log file (default: score_log.json)",
    )
    parser.add_argument(
        "--reset-score",
        action="store_true",
        help="Start score at 0 instead of resuming from the last logged score",
    )
    args = parser.parse_args()

    if args.top_k < 1:
        parser.error("--top-k must be at least 1")

    existing_log = load_score_log(args.log_file)
    if args.reset_score or not existing_log:
        score = 0
    else:
        score = existing_log[-1].get("score", 0)

    model, tokenizer = load_model(args.model, corpus_path=args.corpus)

    print(f"Welcome to Perplexity Explorer! (model={args.model}, top_k={args.top_k})")
    print(f"Persistent score log: {args.log_file} | Starting score: {score}")
    print("Type 'exit' to quit.\n")

    current_seed = None
    while True:
        if current_seed:
            prompt_msg = f"Enter a sentence/phrase (or press Enter to use '{current_seed}'): "
        else:
            prompt_msg = "Enter a sentence or short phrase: "

        user_input = input(prompt_msg).strip()
        if not user_input and current_seed:
            user_input = current_seed
        current_seed = None

        if user_input.lower() in {"exit", "quit"}:
            print(f"Final score: {score}")
            break
        if not user_input:
            continue

        # Optional manual word edit before prediction
        edit_choice = input("Edit any words in this input before predicting? (y/n): ").strip().lower()
        if edit_choice == "y":
            edited = input(f"Enter revised sentence [{user_input}]: ").strip()
            if edited:
                user_input = edited

        candidates, probs = predict_next(model, tokenizer, user_input, top_k=args.top_k)
        print(f"Top-{args.top_k} next-word candidates:")
        for i, (tok, p) in enumerate(zip(candidates, probs), 1):
            print(f"  {i}. {tok} ({p:.4f})")

        answer = input(
            "What is the actual next word? (or type 'i don't know' / 'i am not sure'): "
        ).strip()

        score, delta, status_msg = update_score(score, answer, candidates[0])
        delta_str = f"+{delta}" if delta > 0 else str(delta)
        print(f"{status_msg} (Change: {delta_str} | Current score: {score})")

        if answer.lower() in UNSURE_ANSWERS:
            full_sentence = user_input
        else:
            full_sentence = f"{user_input} {answer}"

        ppl = compute_perplexity(model, tokenizer, full_sentence)
        print(f"Perplexity of '{full_sentence}': {ppl:.2f}")

        # Let the built-in n-gram model learn from the user's new sentence/phrase
        if isinstance(model, NGramLanguageModel):
            model.train_sentence(full_sentence)

        append_score_log(
            args.log_file,
            {
                "model": args.model,
                "input": user_input,
                "top_candidates": [
                    {"word": w, "probability": round(p, 6)} for w, p in zip(candidates, probs)
                ],
                "top1_prediction": candidates[0],
                "user_answer": answer,
                "score_change": delta,
                "score": score,
                "perplexity": round(ppl, 4),
            },
        )
        print(f"Saved round to {args.log_file}.\n")

        gen = input("Generate a new sentence/phrase from the model? (y/n): ").strip().lower()
        if gen == "y":
            generated = generate_sentence(model, tokenizer, full_sentence)
            print(f"Generated: {generated}")
            reuse = input("Use this generated sentence as the next input? (y/n): ").strip().lower()
            if reuse == "y":
                current_seed = generated
        print("---")


if __name__ == "__main__":
    main()
