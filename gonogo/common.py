"""Shared helpers for the Attention Basin multi-candidate go/no-go experiment."""
import json
import re

GEN_SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."

MODELS = {
    "qwen3-8b": "/data/home/202412012250/models/Qwen3-8B",
    "qwen3-4b": "/data/home/202412012250/models/Qwen3-4B",
}


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def dump_jsonl(rows, path):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def last_boxed(text):
    """Content of the last \\boxed{...} (balanced braces), or None."""
    if text is None:
        return None
    idx = max(text.rfind("\\boxed"), text.rfind("\\fbox"))
    if idx < 0:
        return None
    i = text.find("{", idx)
    if i < 0:
        return None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1 : j].strip()
    return None


def norm_key(ans):
    """Cheap string normalisation used only to group identical answers."""
    if ans is None:
        return None
    s = ans
    for t in ["\\left", "\\right", "\\!", "\\,", "\\;", "\\ ", "$", " ", "\\displaystyle", "^\\circ", "^{\\circ}", "\\%", "%"]:
        s = s.replace(t, "")
    s = s.replace("dfrac", "frac").replace("tfrac", "frac")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    return s.rstrip(".")


# ---------------------------------------------------------------------------
# Selection / aggregation prompts
#   mode A: "Solution k:" headers, answer = index of best solution
#   mode B: "Solution k:" headers, answer = final answer (aggregation)
#   mode C: no headers (blank-line separated), answer = final answer
# ---------------------------------------------------------------------------

def build_user_content(problem, cands, mode):
    """Return (content, spans). spans: list of (name, start, end) char offsets in content."""
    N = len(cands)
    spans = []
    buf = []
    pos = 0

    def add(name, text):
        nonlocal pos
        buf.append(text)
        spans.append((name, pos, pos + len(text)))
        pos += len(text)

    add("prefix", f"Problem:\n{problem.strip()}\n\nBelow are {N} candidate solutions to this problem. Some of them may be incorrect.\n\n")
    for k, c in enumerate(cands, 1):
        add(f"h{k}", f"Solution {k}:\n" if mode in ("A", "B") else "")
        add(f"c{k}", c.strip() + "\n\n")
    if mode == "A":
        inst = (f"Which solution is correct? Reply with only the number of the best solution "
                f"in the form \\boxed{{k}}, where k is an integer from 1 to {N}.")
    else:
        inst = ("Based on the candidate solutions above, what is the correct final answer to the problem? "
                "Reply with only the final answer in the form \\boxed{answer}.")
    add("inst", inst)
    return "".join(buf), spans


# Non-thinking Qwen3 ignores "reply with only ..." and starts an analysis, so the
# direct-answer conditions pre-fill the start of the assistant turn.
PREFILL = {"A": "The best solution is Solution ",
           "B": "The final answer is $\\boxed{",
           "C": "The final answer is $\\boxed{"}


def build_chat(tok, user_content, enable_thinking=False, prefill=""):
    return tok.apply_chat_template(
        [{"role": "user", "content": user_content}],
        tokenize=False, add_generation_prompt=True, enable_thinking=enable_thinking,
    ) + prefill


def parse_index(text, N):
    b = last_boxed(text)
    for s in ([b] if b is not None else []) + [text]:
        m = re.search(r"\d+", s or "")
        if m:
            k = int(m.group())
            if 1 <= k <= N:
                return k
    return None


def parse_prefilled_boxed(text):
    """Text generated after a pre-filled '\\boxed{': return content up to the matching brace."""
    depth = 1
    for j, ch in enumerate(text):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[:j].strip()
    return None
