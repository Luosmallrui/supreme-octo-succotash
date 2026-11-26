"""Derive matched E3 readout and heading conditions from the original E2 layouts."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re

from common import dump_jsonl, load_jsonl


def whitespace(text):
    return " ".join(text.split())


def excerpts(text, others):
    """Find short, disjoint passages using only the text of the displayed candidates."""
    tokens = list(re.finditer(r"\S+", text))
    seed = int.from_bytes(hashlib.blake2b(text.encode(), digest_size=8).digest(), "big")
    starts = list(range(len(tokens)))
    random.Random(seed).shuffle(starts)
    chosen = []
    for prose_only in (True, False):
        for width in (8, 6, 12, 16, 4):
            for i in starts:
                if i + width > len(tokens):
                    continue
                lo, hi = tokens[i].start(), tokens[i + width - 1].end()
                if any(lo < end and start < hi for start, end in chosen):
                    continue
                phrase = text[lo:hi]
                if len(phrase) < 16 or len(phrase) > 240:
                    continue
                if prose_only and not re.fullmatch(r"[A-Za-z][A-Za-z ,'’\-.]*[A-Za-z.,]", phrase):
                    continue
                if "boxed" in phrase or "final answer" in phrase.lower():
                    continue
                if any(phrase in other for other in others) or text.count(phrase) != 1:
                    continue
                chosen.append((lo, hi))
                if len(chosen) == 2:
                    return tuple(text[a:b] for a, b in chosen)
    return (text[slice(*chosen[0])] if chosen else "", "")


def prioritize_preview(rows):
    """Exercise all formats early, then keep adjacent layouts together for prefix reuse."""
    preview = []
    used = set()
    kinds = [("label", "numbered"), ("quote", "numbered"), ("answer", "numbered"),
             ("answer", "plain"), ("answer", "none"), ("lookup_quote", "numbered"),
             ("solo_answer", "none")]
    for family in (True, False):
        for readout, header in kinds:
            options = (r for r in rows if (r["readout"], r["header"]) == (readout, header))
            row = next((r for r in options if (r["source"] == "math500") == family), None)
            if row and len(preview) < 12:
                preview.append(row)
                used.add(row["request_id"])
    return preview + [r for r in rows if r["request_id"] not in used]


def make_manifest(sets, e2, graded, problems, model):
    sets8 = [s for s in sets if s["N"] == 8]
    by_set = {s["set_id"]: s for s in sets8}
    if not sets8 or len(by_set) != len(sets8) or len({s["pid"] for s in sets8}) != len(sets8):
        raise ValueError("Expected one unique N=8 set per problem")
    grades = {(g["pid"], g["k"]): g["text"] for g in graded}
    gold = {p["pid"]: p["answer"] for p in problems}
    layouts = defaultdict(list)
    for r in e2:
        if r["model"] != model or r["N"] != 8:
            raise ValueError(f"Unexpected model or size in E2 manifest: {r['request_id']}")
        if r["task"] == "select":
            layouts[r["set_id"]].append(r)
    if by_set.keys() != layouts.keys():
        raise ValueError(f"E1/E2 set mismatch: {sorted(by_set.keys() ^ layouts.keys())}")

    rows = []
    lookup_pool = []
    missing = []
    for sid in sorted(by_set):
        s = by_set[sid]
        pid = s["pid"]
        group = sorted(layouts[sid], key=lambda r: r["request_id"])
        strata = ["one_correct_pair"]
        if s["n_correct"] >= 2:
            strata.append("both_correct_pair")
        expected = {(stratum, pos, label) for stratum in strata for pos in (False, True)
                    for label in (False, True)}
        if len(group) != len(expected) or {(r["stratum"], r["a_first"], r["a_label1"]) for r in group} != expected:
            raise ValueError(f"Incomplete E2 factorial for {sid}")

        texts = {str(k): whitespace(grades[pid, k]) for k in s["ks"]}
        anchors, cues = {}, {}
        for k, text in texts.items():
            anchor, cue = excerpts(text, [v for key, v in texts.items() if key != k])
            anchors[k] = anchor
            if cue:
                cues[k] = cue
            if not anchor:
                missing.append((sid, k))

        for layout in group:
            if (layout["pid"] != pid or set(layout["order"]) != set(s["ks"])
                    or layout["cor"] != s["cor"] or sorted(layout["labels"]) != list(range(1, 9))):
                raise ValueError(f"E1/E2 candidate mismatch for {layout['request_id']}")
            base = {k: v for k, v in layout.items() if k not in ("task", "lookup_slot")}
            base.update(anchors=anchors, anchor_cues=cues, gold=gold[pid], lookup_candidate=None)
            def add(readout, header):
                rid = f"{layout['request_id']}/{readout}/{header}"
                row = dict(base, readout=readout, header=header, request_id=rid)
                rows.append(row)
                return row

            for readout in ("label", "quote", "answer"):
                add(readout, "numbered")
            if layout["stratum"] == "one_correct_pair":
                lookup_pool.append(base)
                if layout["a_label1"]:
                    add("answer", "plain")
                    add("answer", "none")
                    if layout["a_first"]:
                        add("solo_answer", "none")

    # Sample a content-lookup control on 32 distinct problems, four per target
    # physical slot. Candidate correctness is never used to select the cue.
    rng = random.Random(20261126)
    rng.shuffle(lookup_pool)
    used_pids = set()
    slots = list(range(8)) * 4
    rng.shuffle(slots)
    for slot in slots:
        base = next((r for r in lookup_pool if r["pid"] not in used_pids
                     and str(r["order"][slot]) in r["anchor_cues"]), None)
        if base is None:
            raise ValueError(f"Only {len(used_pids)} distinct content-lookup controls available")
        used_pids.add(base["pid"])
        rows.append(dict(base, readout="lookup_quote", header="numbered",
                         lookup_candidate=base["order"][slot],
                         request_id=base["request_id"] + "/lookup_quote/numbered"))

    rows.sort(key=lambda r: (r["set_id"], r["stratum"], r["a_first"], r["a_label1"],
                             r["header"], r["readout"]))
    if len({r["request_id"] for r in rows}) != len(rows):
        raise ValueError("Repeated request IDs")
    print(f"{model}: {len(sets8)} problems, {len(rows)} requests. "
          f"Candidate excerpts unavailable for {len(missing)}/{sum(s['N'] for s in sets8)}: {missing}")
    print(dict(sorted(Counter((r["readout"], r["header"]) for r in rows).items())))
    return prioritize_preview(rows)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    for option in ("sets", "e2-manifest", "graded", "problems", "out"):
        ap.add_argument("--" + option, required=True, type=Path)
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    rows = make_manifest(load_jsonl(args.sets), load_jsonl(args.e2_manifest),
                         load_jsonl(args.graded), json.loads(args.problems.read_text()), args.model)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    dump_jsonl(rows, args.out)
