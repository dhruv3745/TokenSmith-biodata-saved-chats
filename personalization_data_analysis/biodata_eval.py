# biodata_eval.py
from pathlib import Path
import math
from llama_cpp import Llama

# --- Config ---------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
BIODATA_DIR = ROOT / "Biodata_data"
MODEL_PATH = ROOT.parent / "models" / "qwen2.5-1.5b-instruct-q5_k_m.gguf"

PERSONAS = ["p1", "p2", "p3"]
BASELINE_FILE    = BIODATA_DIR / "pi_response_no_biodata.txt"
RESPONSE_PATTERN = "{p}_response_yes_biodata.txt"
BIODATA_PATTERN  = "{p}_biodata.md"

CONTEXT_SIZE   = 4096
THRESHOLD_PCT  = 25.0   # lower perplexity is better, so threshold is on the *decrease*
# -------------------------------------------------------------------------


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
    print(f"Loading model: {MODEL_PATH.name}")
    return Llama(
        model_path=str(MODEL_PATH),
        n_ctx=CONTEXT_SIZE,
        logits_all=True,   # required so we can pull per-token logprobs
        verbose=False,
    )


def conditional_perplexity(llm: Llama, prefix: str, target: str) -> float:
    """
    Perplexity of `target` conditioned on `prefix`.

    Concatenates [prefix + target], runs a single forward pass, then averages
    the negative log-likelihood over only the target tokens (skipping the
    prefix tokens, which we don't want to score).
    """
    prefix_ids = llm.tokenize(prefix.encode("utf-8"), add_bos=True)
    target_ids = llm.tokenize(target.encode("utf-8"), add_bos=False)
    full_ids   = prefix_ids + target_ids

    if len(full_ids) > CONTEXT_SIZE:
        # truncate prefix from the left so target stays intact
        overflow = len(full_ids) - CONTEXT_SIZE
        prefix_ids = prefix_ids[overflow:]
        full_ids   = prefix_ids + target_ids
        print(f"  (truncated prefix by {overflow} tokens to fit context)")

    llm.reset()
    llm.eval(full_ids)

    # llm.scores is shape [n_tokens, vocab_size]; row i = logits predicting token i+1
    import numpy as np
    scores = np.array(llm.scores[: len(full_ids)])

    # We want log P(target_id[t] | everything before it).
    # The logits at position (len(prefix_ids) + t - 1) predict target_ids[t].
    start = len(prefix_ids) - 1   # logits row that predicts the first target token
    end   = start + len(target_ids)
    target_logits = scores[start:end]   # shape [len(target_ids), vocab_size]

    # Stable log-softmax
    max_logits = target_logits.max(axis=-1, keepdims=True)
    log_probs = target_logits - max_logits - np.log(
        np.exp(target_logits - max_logits).sum(axis=-1, keepdims=True)
    )

    token_logprobs = log_probs[np.arange(len(target_ids)), target_ids]
    avg_neg_logprob = -token_logprobs.mean()
    return float(math.exp(avg_neg_logprob))


def read(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return path.read_text(encoding="utf-8")


def main():
    llm = load_model()
    baseline_text = read(BASELINE_FILE)

    header = f"{'Persona':<8} {'PPL No Bio':>11} {'PPL With Bio':>13} {'% Diff':>9}  Significant?"
    print("\n" + header)
    print("-" * len(header))

    for p in PERSONAS:
        bio_text  = read(BIODATA_DIR / BIODATA_PATTERN.format(p=p))
        with_text = read(BIODATA_DIR / RESPONSE_PATTERN.format(p=p))

        # PPL of each response given the persona's biodata as prefix.
        # Lower PPL = biodata better predicts the response = stronger alignment.
        ppl_baseline = conditional_perplexity(llm, bio_text, baseline_text)
        ppl_with     = conditional_perplexity(llm, bio_text, with_text)

        # % decrease (positive = improvement, since lower PPL is better)
        pct = ((ppl_baseline - ppl_with) / ppl_baseline * 100) if ppl_baseline else 0.0
        flag = "yes" if pct >= THRESHOLD_PCT else "no"

        print(f"{p:<8} {ppl_baseline:>11.2f} {ppl_with:>13.2f} {pct:>8.1f}%  {flag}")


if __name__ == "__main__":
    main()