"""E3 readout, heading, and question-only controls on a fixed candidate layout."""
import argparse
import fcntl
import json
import os
import re
import sys
from functools import lru_cache
from pathlib import Path

from common import MODELS, build_chat, load_jsonl
from e2_run import recover_results


HERE = Path(__file__).resolve().parent
BUDGET = {"direct": 256, "think": 12288}
FORMAT = '\nGive the response on the final line in the format FINAL: <response>.'
FINAL = re.compile(r"^[ \t]*FINAL:[ \t]*", re.MULTILINE)


def whitespace(text):
    return " ".join(text.split())


def user_prompt(row, problem, candidates):
    problem_line = f"Problem:\n{problem.strip()}"
    readout = row["readout"]
    if readout == "solo_answer":
        return problem_line + "\n\nSolve the problem and return its final mathematical answer." + FORMAT

    order, labels, n = row["order"], row["labels"], row["N"]
    if len(order) != n or len(set(order)) != n or sorted(labels) != list(range(1, n + 1)):
        raise ValueError(f"Invalid layout for {row['request_id']}")
    chunks = [problem_line,
              f"Below are {n} candidate solutions to this problem. Some of them may be incorrect."]
    for k, label in zip(order, labels):
        if row["header"] == "numbered":
            header = f"Solution {label}:\n"
        elif row["header"] == "plain":
            header = "-----\n"
        elif row["header"] == "none":
            header = ""
        else:
            raise ValueError(f"Invalid header for {row['request_id']}")
        text = candidates[row["pid"], k].strip()
        chunks.append(header + text)
    if readout == "label" and row["header"] == "numbered":
        instruction = ("Choose the best solution. Return only its printed Solution number. "
                       "The printed numbers may appear in any order.")
    elif readout == "quote":
        instruction = ("Choose the best solution. Copy a short passage verbatim from its text that uniquely "
                       "identifies it (aim for 6 to 16 words). Include some of its working, not just its final answer.")
    elif readout == "lookup_quote":
        cue = whitespace(row["anchor_cues"][str(row["lookup_candidate"])])
        instruction = (f"Locate the solution that contains this passage verbatim:\n{cue}\n"
                       "Copy a different, non-overlapping passage verbatim from that solution that uniquely "
                       "identifies it (aim for 6 to 16 words). Include some of its working, not just its final answer.")
    elif readout == "answer":
        instruction = "Use the candidate solutions to determine the final mathematical answer to the problem."
    else:
        raise ValueError(f"Unsupported readout/heading for {row['request_id']}")
    return "\n\n".join(chunks + [instruction + FORMAT])


def final_response(generated, mode):
    if "</think>" in generated:
        generated = generated.rsplit("</think>", 1)[1]
    elif mode == "think" or "<think>" in generated:
        return None
    matches = list(FINAL.finditer(generated))
    if not matches:
        return None
    return generated[matches[-1].end():].strip() or None


@lru_cache(maxsize=1024)
def parse_gold(gold):
    from math_verify import parse
    parsed = parse("$" + gold + "$")
    if not parsed:
        raise ValueError(f"Unparseable reference answer {gold!r}")
    return parsed


def grade_math(final, gold):
    from math_verify import parse, verify
    expected = parse_gold(gold)
    text = final.strip()
    for left, right in [("$$", "$$"), ("$", "$"), (r"\(", r"\)"), (r"\[", r"\]")]:
        if text.startswith(left) and text.endswith(right) and len(text) > len(left) + len(right):
            text = text[len(left):-len(right)].strip()
            break
    try:
        candidate = parse(text if text.startswith(r"\boxed{") else r"\boxed{" + text + "}")
        return bool(candidate), bool(candidate and verify(expected, candidate))
    except Exception:
        # Preserve the generation and count a failed parse as a failed trial.
        return False, False


QUOTE_WRAPPERS = [('"', '"'), ("'", "'"), ("“", "”"), ("‘", "’"), ("«", "»"), ("`", "`")]


def offsets(text, piece):
    start = text.find(piece)
    while start != -1:
        yield start
        start = text.find(piece, start + 1)


def match_quote(final, row, candidates):
    excerpt = whitespace(final)
    for left, right in QUOTE_WRAPPERS:
        if len(excerpt) >= 2 and excerpt.startswith(left) and excerpt.endswith(right):
            excerpt = excerpt[1:-1].strip()
            break
    if len(excerpt) < 16 or len(excerpt.split()) < 4:
        return None, "too_short"
    bodies = [whitespace(candidates[row["pid"], k]) for k in row["order"]]
    slots = [i for i, body in enumerate(bodies) if excerpt in body]
    if not slots:
        return None, "absent"
    if len(slots) != 1:
        return None, "ambiguous"
    slot = slots[0]
    if row["readout"] == "lookup_quote" and row["order"][slot] == row["lookup_candidate"]:
        cue = whitespace(row["anchor_cues"][str(row["lookup_candidate"])])
        if not cue or cue not in bodies[slot]:
            raise ValueError(f"Missing lookup cue in original text for {row['request_id']}")
        cue_starts = list(offsets(bodies[slot], cue))
        independent = any(all(q + len(excerpt) <= c or q >= c + len(cue) for c in cue_starts)
                          for q in offsets(bodies[slot], excerpt))
        if not independent:
            return None, "overlaps_cue"
    return slot, None


def result(row, mode, output, n_prompt, candidates, grader=grade_math):
    final = final_response(output.text, mode)
    candidate = slot = label = None
    parsed = correct = False
    error = "missing_final" if final is None else None
    if final is not None:
        if row["readout"] == "label":
            error = "invalid_label"
            if re.fullmatch(r"[1-9][0-9]*", final) and len(final) <= len(str(row["N"])):
                number = int(final)
                if number in row["labels"]:
                    slot = row["labels"].index(number)
                    candidate = row["order"][slot]
                    error = None
        elif row["readout"] in ("quote", "lookup_quote"):
            slot, error = match_quote(final, row, candidates)
            if slot is not None:
                candidate = row["order"][slot]
        else:
            parsed, correct = grader(final, row["gold"])
            if not parsed:
                error = "invalid_math_answer"
        if slot is not None:
            parsed = True
            label = row["labels"][slot]
            correct = (candidate == row["lookup_candidate"] if row["readout"] == "lookup_quote"
                       else bool(row["cor"][str(candidate)]))
    return dict(row, mode=mode, choice_candidate=candidate, choice_slot=slot + 1 if slot is not None else None,
                choice_label=label, correct=correct, parsed=parsed, parse_error=error, final=final, gen=output.text,
                finish=output.finish_reason, n_out=len(output.token_ids), n_prompt=n_prompt,
                max_tokens=BUDGET[mode])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--model", choices=MODELS, required=True)
    ap.add_argument("--mode", choices=BUDGET, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()
    if args.batch_size < 1:
        ap.error("--batch-size must be positive")
    rows = [r for r in load_jsonl(args.manifest) if r["model"] == args.model]
    wanted = {r["request_id"]: r for r in rows}
    if not rows or len(wanted) != len(rows):
        raise ValueError("No rows for this model or duplicate request IDs")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("a+b") as sink:
        fcntl.flock(sink, fcntl.LOCK_EX | fcntl.LOCK_NB)
        done = recover_results(sink, wanted, args.mode)
        pending = [r for r in rows if r["request_id"] not in done]
        print(f"{args.model} {args.mode}: {len(done)}/{len(rows)} already saved", flush=True)
        if not pending:
            return

        sys.path.insert(0, str(HERE / "pylib"))
        probs = {p["pid"]: p["problem"] for p in json.loads((HERE / "data/problems.json").read_text())}
        cands = {(r["pid"], r["k"]): r["text"]
                 for r in load_jsonl(HERE / "out" / f"graded_{args.model}.jsonl")}
        # Resolve math grading before allocating the GPU.
        for gold in {r["gold"] for r in pending if r["readout"] in ("answer", "solo_answer")}:
            parse_gold(gold)
        from vllm import LLM, SamplingParams

        llm = LLM(model=MODELS[args.model], max_model_len=40960, gpu_memory_utilization=0.92,
                  enable_prefix_caching=True, max_num_batched_tokens=32768, seed=0)
        tok = llm.get_tokenizer()
        sampling = SamplingParams(temperature=0, max_tokens=BUDGET[args.mode])
        for start in range(0, len(pending), args.batch_size):
            chunk = pending[start:start + args.batch_size]
            prompts = [build_chat(tok, user_prompt(r, probs[r["pid"]], cands), enable_thinking=args.mode == "think")
                       for r in chunk]
            outputs = llm.generate(prompts, sampling, use_tqdm=False)
            for r, o in zip(chunk, outputs, strict=True):
                record = result(r, args.mode, o.outputs[0], len(o.prompt_token_ids), cands)
                sink.write((json.dumps(record, ensure_ascii=False) + "\n").encode())
            sink.flush()
            os.fsync(sink.fileno())
            print(f"Saved {len(done) + start + len(chunk)}/{len(rows)}", flush=True)


if __name__ == "__main__":
    main()
