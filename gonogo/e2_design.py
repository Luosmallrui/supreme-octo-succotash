"""Cross physical position and printed label within a fixed eight-solution set."""
import argparse
from collections import Counter
from itertools import product
from pathlib import Path
import random

from common import dump_jsonl, load_jsonl


def make_manifest(sets, model, seed):
    rng = random.Random(seed)
    sets = sorted((s for s in sets if s["N"] == 8), key=lambda s: s["set_id"])
    if not sets or len({s["pid"] for s in sets}) != len(sets):
        raise ValueError("Expected exactly one N=8 candidate set per included problem")
    rows = []
    for stratum in ("one_correct_pair", "both_correct_pair"):
        eligible = [s for s in sets if stratum == "one_correct_pair" or s["n_correct"] >= 2]
        rng.shuffle(eligible)
        # Counterbalance the comparison position across problems in each stratum.
        slots = [2 + i % 7 for i in range(len(eligible))]
        rng.shuffle(slots)
        for s, j in zip(eligible, slots):
            good = [k for k in s["ks"] if s["cor"][str(k)]]
            bad = [k for k in s["ks"] if not s["cor"][str(k)]]
            if stratum == "one_correct_pair":
                a, b = rng.choice(good), rng.choice(bad)
            else:
                a, b = rng.sample(good, 2)
            other = [k for k in s["ks"] if k not in (a, b)]
            rng.shuffle(other)
            spare_labels = list(range(2, 9))
            rng.shuffle(spare_labels)
            alternate = spare_labels.pop()
            pair_seed = rng.randrange(2**30)
            block = []
            for a_first, a_label1 in product((False, True), repeat=2):
                order = other.copy()
                order.insert(0, a if a_first else b)
                order.insert(j - 1, b if a_first else a)
                label_of = dict(zip(other, spare_labels))
                label_of[a], label_of[b] = (1, alternate) if a_label1 else (alternate, 1)
                prefix = f"{s['set_id']}/{stratum}/{int(a_first)}{int(a_label1)}"
                block.append(dict(model=model, set_id=s["set_id"], pid=s["pid"], source=s["source"],
                                  N=8, stratum=stratum, task="select", probe_slot=j, a_first=a_first,
                                  a_label1=a_label1, order=order, labels=[label_of[k] for k in order],
                                  a=a, b=b, cor=s["cor"], lookup_slot=None, request_id=prefix,
                                  seed=pair_seed))
            rows.extend(block)

    # 32 reading controls on different problems. Each printed label is the target
    # four times and always differs from the ordinal position that is queried.
    used = set()
    candidates = [r for r in rows if r["stratum"] == "one_correct_pair"]
    for i in range(32):
        target = 1 + i % 8
        choices = [r for r in candidates if r["pid"] not in used and r["labels"].index(target) != target - 1]
        example = rng.choice(choices)
        used.add(example["pid"])
        rows.append(dict(example, task="lookup", lookup_slot=example["labels"].index(target) + 1,
                         request_id=example["request_id"] + "/lookup"))
    rng.shuffle(rows)
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sets", required=True, type=Path)
    ap.add_argument("--model", required=True)
    ap.add_argument("--seed", type=int, default=20261125)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    manifest = make_manifest(load_jsonl(args.sets), args.model, args.seed)
    dump_jsonl(manifest, args.out)
    print(f"{args.model}: {len(manifest)} requests, {Counter((r['task'], r['stratum']) for r in manifest)}")
