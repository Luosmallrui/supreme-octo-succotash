"""Pre-filled label readout shared by E5 and E6 (same as e1_run.py direct mode).

P(label = k) for k = 1..n: first-token logprobs give P(first digit d); for n >= 10 a second pass on prefix + d gives
P(next digit | d), so P(10d + x) = P(d) P(x | d) and P(d alone) = P(d) (1 - sum_x P(x | d)).
"""
import math


def digit_probs(lp):
    """vLLM logprob dict -> {digit string: prob} for single-digit tokens."""
    out = {}
    for v in lp.values():
        t = (v.decoded_token or "").strip()
        if len(t) == 1 and t.isascii() and t.isdigit():
            out[t] = out.get(t, 0.0) + math.exp(v.logprob)
    return out


def combine(first, second, n):
    """Distribution over labels 1..n from first-digit and second-digit probabilities; returns (p, raw mass)."""
    p = [0.0] * n
    for d in range(1, 10):
        pd = first.get(str(d), 0.0)
        if d > n or pd == 0:
            continue
        nxt = second.get(d, {})
        for x in range(10):
            if 10 * d + x <= n:
                p[10 * d + x - 1] += pd * nxt.get(str(x), 0.0)
        p[d - 1] += pd * (1 - sum(nxt.values()))
    mass = sum(p)
    return ([x / mass for x in p] if mass > 0 else p), mass


def score(llm, prefixes, ns, chunk=8, log=None):
    """Label distributions for pre-filled prompts. Returns list of (p, mass, n_prompt_tokens)."""
    from vllm import SamplingParams
    sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)
    out = [None] * len(prefixes)
    for c0 in range(0, len(prefixes), chunk):
        idx = list(range(c0, min(c0 + chunk, len(prefixes))))
        o1 = llm.generate([prefixes[i] for i in idx], sp, use_tqdm=False)
        first = {i: digit_probs(o.outputs[0].logprobs[0]) for i, o in zip(idx, o1)}
        ntok = {i: len(o.prompt_token_ids) for i, o in zip(idx, o1)}
        second = [(i, d) for i in idx for d in range(1, ns[i] // 10 + 1)]
        nxt = {i: {} for i in idx}
        if second:
            o2 = llm.generate([prefixes[i] + str(d) for i, d in second], sp, use_tqdm=False)
            for (i, d), o in zip(second, o2):
                nxt[i][d] = digit_probs(o.outputs[0].logprobs[0])
        for i in idx:
            p, mass = combine(first[i], nxt[i], ns[i])
            out[i] = (p, mass, ntok[i])
        if log and (c0 // chunk) % 50 == 0:
            log(f"scored {idx[-1] + 1}/{len(prefixes)}")
    return out
