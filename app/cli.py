"""Command-line entry point: `python -m app.cli --help`."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

from app import llm
from app.core import (
    DEFAULT_CHECKPOINT_PATH,
    BytePairTokenizer,
    TrainingConfig,
    build_quiz_question,
    corpus_stats,
    ensure_trained_model,
    generate_completion,
    get_corpus_text,
    load_checkpoint,
    search_corpus,
    train_model,
)
from app.storage import Store


def _supports_color() -> bool:
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _style(text: str, code: str) -> str:
    if not _supports_color():
        return text
    return f"\033[{code}m{text}\033[0m"


def _cmd_train(args: argparse.Namespace) -> None:
    cfg = TrainingConfig(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        max_seq_len=args.max_seq_len,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.lr,
        stride=args.stride,
        device=args.device,
    )
    print(_style("Training MedicalGPT from scratch on curated medical corpus...", "1;36"))

    def on_epoch(rec: dict) -> None:
        if rec["epoch"] == 1 or rec["epoch"] % max(1, cfg.epochs // 5) == 0 or rec["epoch"] == cfg.epochs:
            print(
                f"  Epoch {rec['epoch']:>3}/{cfg.epochs} | "
                f"train_loss={rec['train_loss']:.4f} | "
                f"val_loss={rec['val_loss']:.4f} | "
                f"ppl={rec['perplexity']:.2f}"
            )

    result = train_model(
        config=cfg,
        checkpoint_path=args.checkpoint,
        progress_callback=on_epoch,
    )

    store = Store()
    store.put("last_training", {
        "epochs": cfg.epochs,
        "vocab_size": result["tokenizer"].vocab_size,
        "num_parameters": result["num_parameters"],
        "initial_loss": result["initial_loss"],
        "final_loss": result["final_loss"],
        "final_val_loss": result["final_val_loss"],
        "final_perplexity": result["final_perplexity"],
        "elapsed_seconds": result["elapsed_seconds"],
        "checkpoint_path": result["checkpoint_path"],
    })

    print(
        _style(
            f"\nDone in {result['elapsed_seconds']}s | "
            f"params={result['num_parameters']:,} | "
            f"loss {result['initial_loss']:.4f} -> {result['final_loss']:.4f} | "
            f"saved to {result['checkpoint_path']}",
            "1;32",
        )
    )


def _load_or_train(checkpoint: str, fast_auto_train: bool = True):
    ckpt_path = Path(checkpoint)
    if ckpt_path.exists():
        return load_checkpoint(ckpt_path)

    print(_style(f"No checkpoint found at {ckpt_path}. Training a starter MedicalGPT model...", "33"))
    cfg = TrainingConfig(epochs=20 if fast_auto_train else 30)
    return ensure_trained_model(checkpoint_path=ckpt_path, config=cfg)


def _cmd_complete(args: argparse.Namespace) -> None:
    prompt = " ".join(args.prompt)
    model, tokenizer, _ = _load_or_train(args.checkpoint)

    result = generate_completion(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
    )

    print(_style("\nMedicalGPT Completion:", "1;36"))
    print(f"  {_style(result['prompt'], '1;37')}{_style(result['completion'], '32')}")
    print(
        _style(
            f"  [prompt_tokens={result['prompt_tokens']}, generated_tokens={result['generated_tokens']}, "
            f"temp={args.temperature}, top_k={args.top_k}]",
            "90",
        )
    )

    if args.reference:
        matches = search_corpus(prompt.split()[0], limit=2) if prompt.strip() else []
        if matches:
            print(_style("\nCorpus Reference:", "1;35"))
            for m in matches:
                print(f"  [{m['category']}] {m['text']}")

    if args.explain:
        try:
            explanation = llm.chat([
                {"role": "system", "content": "You are a concise medical terminology tutor."},
                {"role": "user", "content": f"Briefly explain the medical concept in: {result['full_text']}"},
            ])
            print(_style("\nTutor Explanation (LLM):", "1;34"))
            print(f"  {explanation.strip()}")
        except Exception as exc:
            print(_style(f"\n[LLM explanation unavailable: {exc}]", "33"))


def _cmd_lookup(args: argparse.Namespace) -> None:
    query = " ".join(args.query)
    matches = search_corpus(query, limit=args.limit)
    if not matches:
        print(f"No medical terms matched '{query}'.")
        return
    print(_style(f"Found {len(matches)} matching medical entries for '{query}':", "1;36"))
    for i, entry in enumerate(matches, 1):
        print(f"  {i}. [{entry['category']}] {entry['text']}")


def _cmd_inspect(args: argparse.Namespace) -> None:
    stats = corpus_stats()
    print(_style("Medical Corpus Summary:", "1;36"))
    print(f"  Entries:    {stats['num_entries']}")
    print(f"  Categories: {stats['num_categories']} ({', '.join(stats['categories'][:4])}...)")
    print(f"  Words:      {stats['num_words']:,}")
    print(f"  Characters: {stats['num_chars']:,}")

    ckpt_path = Path(args.checkpoint)
    if ckpt_path.exists():
        model, tokenizer, history = load_checkpoint(ckpt_path)
        print(_style("\nTrained MedicalGPT Model:", "1;36"))
        print(f"  Checkpoint: {ckpt_path}")
        print(f"  Vocab size: {tokenizer.vocab_size}")
        print(f"  Parameters: {model.num_parameters():,}")
        print(
            f"  Architecture: d_model={model.config.d_model}, "
            f"n_heads={model.config.n_heads}, n_layers={model.config.n_layers}, "
            f"max_seq_len={model.config.max_seq_len}"
        )
        if history:
            print(f"  Last epoch: {history[-1]['epoch']} (train_loss={history[-1]['train_loss']}, ppl={history[-1]['perplexity']})")
    else:
        tokenizer = BytePairTokenizer().train(get_corpus_text(), vocab_size=400)
        print(_style("\nNo saved checkpoint yet (run `python -m app.cli train`). Showing live BPE tokenizer:", "33"))

    sample = args.text
    pieces = tokenizer.tokenize_with_pieces(sample)
    raw_bytes = len(sample.encode("utf-8"))
    ratio = raw_bytes / max(1, len(pieces))
    print(_style(f"\nBPE Tokenization for: {sample!r}", "1;35"))
    print(f"  Tokens ({len(pieces)} vs {raw_bytes} bytes, {ratio:.2f}x compression):")
    formatted = " | ".join(f"{p['piece']!r}({p['id']})" for p in pieces)
    print(f"  {formatted}")


def _cmd_quiz(args: argparse.Namespace) -> None:
    model, tokenizer, _ = _load_or_train(args.checkpoint)
    for idx in range(1, args.count + 1):
        q = build_quiz_question(seed=args.seed + idx if args.seed is not None else None)
        gpt_out = generate_completion(
            model=model,
            tokenizer=tokenizer,
            prompt=q["prompt"],
            max_new_tokens=28,
            temperature=0.4,
        )
        print(_style(f"\nQuestion {idx}/{args.count} [{q['category']}]", "1;36"))
        print(f"  Prompt:      {q['prompt']}___")
        if not args.non_interactive:
            try:
                user_ans = input("  Your guess (or press Enter to reveal): ").strip()
                if user_ans:
                    print(f"  You wrote:   {q['prompt']}{user_ans}")
            except EOFError:
                pass
        print(f"  MedicalGPT:  {gpt_out['full_text']}")
        print(f"  Reference:   {q['full_text']}")


def _cmd_interactive(args: argparse.Namespace) -> None:
    model, tokenizer, history = _load_or_train(args.checkpoint)
    temp = args.temperature
    top_k = args.top_k
    max_tokens = args.max_tokens

    print(_style("Medical Terminology Helper — Interactive Study CLI", "1;36"))
    print("Type a partial medical phrase to complete it.")
    print("Commands: /lookup <term> | /quiz | /temp <val> | /tokens <n> | /stats | /quit\n")

    while True:
        try:
            line = input(_style("med-gpt> ", "1;35")).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not line:
            continue
        if line in {"/quit", "/exit", "quit", "exit"}:
            print("Goodbye! Keep studying.")
            break
        if line.startswith("/temp "):
            try:
                temp = float(line.split(maxsplit=1)[1])
                print(f"Temperature set to {temp}")
            except ValueError:
                print("Usage: /temp 0.7")
            continue
        if line.startswith("/tokens "):
            try:
                max_tokens = max(1, int(line.split(maxsplit=1)[1]))
                print(f"Max new tokens set to {max_tokens}")
            except ValueError:
                print("Usage: /tokens 36")
            continue
        if line.startswith("/lookup "):
            query = line.split(maxsplit=1)[1]
            matches = search_corpus(query, limit=3)
            if not matches:
                print(f"No entries matched '{query}'.")
            for m in matches:
                print(f"  [{m['category']}] {m['text']}")
            continue
        if line == "/quiz":
            q = build_quiz_question()
            print(f"  [{q['category']}] Complete: {q['prompt']}___")
            try:
                input("  Press Enter to reveal reference & MedicalGPT completion... ")
            except EOFError:
                pass
            gpt_out = generate_completion(model, tokenizer, q["prompt"], max_new_tokens=28, temperature=temp, top_k=top_k)
            print(f"  MedicalGPT: {gpt_out['full_text']}")
            print(f"  Reference:  {q['full_text']}")
            continue
        if line == "/stats":
            print(
                f"  Model params={model.num_parameters():,}, vocab={tokenizer.vocab_size}, "
                f"d_model={model.config.d_model}, layers={model.config.n_layers}, heads={model.config.n_heads}"
            )
            if history:
                print(f"  Training epochs={len(history)}, final_loss={history[-1]['train_loss']}")
            continue

        result = generate_completion(
            model=model,
            tokenizer=tokenizer,
            prompt=line,
            max_new_tokens=max_tokens,
            temperature=temp,
            top_k=top_k,
        )
        print(f"  {result['prompt']}{_style(result['completion'], '32')}\n")


def _cmd_web(args: argparse.Namespace) -> None:
    from app.web import run_server

    run_server(host=args.host, port=args.port, checkpoint_path=args.checkpoint)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="medical-terminology-helper",
        description=(
            "Medical students get a command-line assistant completing medical phrases "
            "and definitions, trained on public domain medical texts to reinforce terminology learning."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("hello", help="check that everything is wired up")

    p_train = sub.add_parser("train", help="train the MedicalGPT model from scratch on the medical corpus")
    p_train.add_argument("--epochs", type=int, default=25, help="number of training epochs (default: 25)")
    p_train.add_argument("--vocab-size", type=int, default=512, help="BPE vocabulary size (default: 512)")
    p_train.add_argument("--d-model", type=int, default=128, help="embedding dimension (default: 128)")
    p_train.add_argument("--n-heads", type=int, default=4, help="attention heads (default: 4)")
    p_train.add_argument("--n-layers", type=int, default=3, help="transformer blocks (default: 3)")
    p_train.add_argument("--max-seq-len", type=int, default=96, help="context window length (default: 96)")
    p_train.add_argument("--batch-size", type=int, default=16, help="training batch size (default: 16)")
    p_train.add_argument("--lr", type=float, default=3e-3, help="learning rate (default: 3e-3)")
    p_train.add_argument("--stride", type=int, default=16, help="sliding window stride (default: 16)")
    p_train.add_argument("--device", default="auto", help="device: auto, cpu, or cuda")
    p_train.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT_PATH), help="path to save model checkpoint")

    p_complete = sub.add_parser("complete", help="complete a partial medical phrase or definition")
    p_complete.add_argument("prompt", nargs="+", help="partial medical phrase to complete")
    p_complete.add_argument("--max-tokens", type=int, default=36, help="maximum tokens to generate")
    p_complete.add_argument("--temperature", type=float, default=0.6, help="sampling temperature (0 for greedy)")
    p_complete.add_argument("--top-k", type=int, default=20, help="top-k sampling cutoff")
    p_complete.add_argument("--top-p", type=float, default=0.9, help="nucleus sampling probability")
    p_complete.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT_PATH), help="model checkpoint path")
    p_complete.add_argument("--reference", action="store_true", help="also print matching corpus reference definitions")
    p_complete.add_argument("--explain", action="store_true", help="add an optional LLM tutor explanation via .env")

    p_lookup = sub.add_parser("lookup", help="search the curated public-domain medical corpus")
    p_lookup.add_argument("query", nargs="+", help="term or keyword to search")
    p_lookup.add_argument("--limit", type=int, default=5, help="maximum results")

    p_inspect = sub.add_parser("inspect", help="inspect BPE tokenization and MedicalGPT architecture")
    p_inspect.add_argument(
        "--text",
        default="Myocardium is the thick muscular middle layer of the heart wall.",
        help="sample medical text to tokenize",
    )
    p_inspect.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT_PATH), help="model checkpoint path")

    p_quiz = sub.add_parser("quiz", help="practice medical terminology with fill-in-the-blank questions")
    p_quiz.add_argument("--count", type=int, default=3, help="number of quiz questions")
    p_quiz.add_argument("--seed", type=int, default=None, help="random seed for reproducible questions")
    p_quiz.add_argument("--non-interactive", action="store_true", help="print questions and answers without prompting")
    p_quiz.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT_PATH), help="model checkpoint path")

    p_repl = sub.add_parser("interactive", help="launch interactive medical terminology study REPL")
    p_repl.add_argument("--max-tokens", type=int, default=36, help="maximum tokens per completion")
    p_repl.add_argument("--temperature", type=float, default=0.6, help="sampling temperature")
    p_repl.add_argument("--top-k", type=int, default=20, help="top-k sampling cutoff")
    p_repl.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT_PATH), help="model checkpoint path")

    p_web = sub.add_parser("web", help="launch the local Medical Terminology Helper web studio")
    p_web.add_argument("--host", default="127.0.0.1", help="host interface (default: 127.0.0.1)")
    p_web.add_argument("--port", type=int, default=8000, help="port number (default: 8000)")
    p_web.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT_PATH), help="model checkpoint path")

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "hello":
        print("Medical Terminology Helper is ready. Open README.md for your milestones.")
    elif args.command == "train":
        _cmd_train(args)
    elif args.command == "complete":
        _cmd_complete(args)
    elif args.command == "lookup":
        _cmd_lookup(args)
    elif args.command == "inspect":
        _cmd_inspect(args)
    elif args.command == "quiz":
        _cmd_quiz(args)
    elif args.command == "interactive":
        _cmd_interactive(args)
    elif args.command == "web":
        _cmd_web(args)


if __name__ == "__main__":
    main()
