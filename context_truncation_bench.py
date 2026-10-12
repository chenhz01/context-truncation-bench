#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""context-truncation-bench — benchmark context-window truncation strategies.

Compares head / tail / middle / head_tail / top_importance / hybrid truncation
strategies on three retention metrics (token, importance-weighted, critical)
over a unit-scored document corpus. Pure stdlib.
"""
import argparse
import json
import random
import sys
import time

WEIGHTS = {
    "system": 1.0,
    "instruction": 0.9,
    "fact": 0.7,
    "example": 0.4,
    "noise": 0.1,
}
DEFAULT_WEIGHT = 0.5
STRATEGIES = ("head", "tail", "middle", "head_tail", "top_importance", "hybrid")


class BenchError(Exception):
    """Recoverable input error surfaced as a CLI usage failure (exit 2)."""


# ---------------------------------------------------------------- corpus model

def token_count(text):
    return len(text.split())


def make_unit(uid, kind, text, critical=False):
    if not isinstance(text, str):
        raise BenchError("unit %r text must be str, got %s" % (uid, type(text).__name__))
    return {
        "id": uid,
        "kind": kind,
        "text": text,
        "tokens": token_count(text),
        "weight": WEIGHTS.get(kind, DEFAULT_WEIGHT),
        "critical": bool(critical),
    }


def unit_weight(unit):
    return unit.get("weight", WEIGHTS.get(unit.get("kind"), DEFAULT_WEIGHT))


def make_corpus(n_units, seed=42, vocab=None):
    """Deterministic synthetic corpus mixing all kinds; every 7th instruction is critical."""
    rnd = random.Random(seed)
    if vocab is None:
        vocab = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta"]
    kinds = ["system", "instruction", "fact", "example", "noise"]
    units = []
    for i in range(n_units):
        kind = kinds[i % len(kinds)]
        words = [rnd.choice(vocab) for _ in range(6 + rnd.randrange(18))]
        critical = kind == "instruction" and (i % 7 == 1)
        units.append(make_unit("u%05d" % i, kind, " ".join(words), critical=critical))
    return units


def load_units(path):
    with open(path, "r", encoding="utf-8") as fh:
        raw = json.load(fh)
    if not isinstance(raw, list):
        raise BenchError("units file must be a JSON list")
    units = []
    for i, item in enumerate(raw):
        units.append(make_unit(
            item.get("id", "u%05d" % i),
            item.get("kind", "fact"),
            item.get("text", ""),
            critical=item.get("critical", False),
        ))
    return units


# ------------------------------------------------------------------ strategies

def _pack(order, budget):
    """Greedy prefix pack of ordered units until budget (tokens) is filled."""
    kept, used = [], 0
    for u in order:
        if used + u["tokens"] > budget and kept:
            break
        if used + u["tokens"] > budget:
            break
        kept.append(u)
        used += u["tokens"]
    return kept


def truncate_head(units, budget):
    return _pack(units, budget)


def truncate_tail(units, budget):
    return list(reversed(_pack(list(reversed(units)), budget)))


def truncate_middle(units, budget):
    """Keep the most centred contiguous window that fits the token budget."""
    n = len(units)
    acc = [0]
    for u in units:
        acc.append(acc[-1] + u["tokens"])
    total = acc[-1]
    if total <= budget:
        return list(units)
    best_key = None
    best_range = None
    lo = 0
    corpus_center = (n - 1) / 2.0
    for hi in range(1, n + 1):
        while lo < hi and acc[hi] - acc[lo] > budget:
            lo += 1
        if lo == hi:  # single oversized unit cannot fit
            continue
        toks = acc[hi] - acc[lo]
        center = (lo + hi - 1) / 2.0
        key = (abs(center - corpus_center), -toks)
        if best_key is None or key < best_key:
            best_key = key
            best_range = (lo, hi)
    if best_range is None:
        return []
    return list(units[best_range[0]:best_range[1]])


def truncate_head_tail(units, budget):
    half = budget // 2
    head = _pack(units, half)
    tail = truncate_tail(units, budget - sum(u["tokens"] for u in head))
    return head + tail


def truncate_top_importance(units, budget):
    order = sorted(range(len(units)),
                   key=lambda i: (units[i]["critical"], unit_weight(units[i]), -i),
                   reverse=True)
    picked = set()
    used = 0
    for i in order:
        u = units[i]
        if used + u["tokens"] <= budget:
            picked.add(i)
            used += u["tokens"]
    return [u for i, u in enumerate(units) if i in picked]


def truncate_hybrid(units, budget):
    """40% head, 40% tail, 20% top-importance of the unpicked remainder."""
    head_budget = int(budget * 0.4)
    tail_budget = int(budget * 0.4)
    rest_budget = budget - head_budget - tail_budget
    head = _pack(units, head_budget)
    tail = truncate_tail(units, tail_budget)
    head_ids = {id(u) for u in head}
    tail_ids = {id(u) for u in tail}
    rest = [u for u in units if id(u) not in head_ids and id(u) not in tail_ids]
    mid = truncate_top_importance(rest, rest_budget)
    return head + mid + tail


STRATEGY_FNS = {
    "head": truncate_head,
    "tail": truncate_tail,
    "middle": truncate_middle,
    "head_tail": truncate_head_tail,
    "top_importance": truncate_top_importance,
    "hybrid": truncate_hybrid,
}


# -------------------------------------------------------------------- metrics

def retention_metrics(all_units, kept):
    total_tokens = sum(u["tokens"] for u in all_units) or 1
    kept_tokens = sum(u["tokens"] for u in kept)
    total_weighted = sum(unit_weight(u) * u["tokens"] for u in all_units) or 1.0
    kept_weighted = sum(unit_weight(u) * u["tokens"] for u in kept)
    total_crit = sum(1 for u in all_units if u["critical"])
    kept_crit = sum(1 for u in kept if u["critical"])
    return {
        "token_retention": round(kept_tokens / total_tokens, 4),
        "weighted_retention": round(kept_weighted / total_weighted, 4),
        "critical_retention": round(kept_crit / total_crit, 4) if total_crit else 1.0,
        "kept_tokens": kept_tokens,
    }


def composite_score(metrics):
    return round(0.6 * metrics["weighted_retention"] + 0.4 * metrics["critical_retention"], 4)


def run_bench(units, budget):
    if not units:
        return {"strategies": {}, "winner": None, "note": "empty corpus"}
    rows = {}
    for name in STRATEGIES:
        kept = STRATEGY_FNS[name](units, budget)
        m = retention_metrics(units, kept)
        m["score"] = composite_score(m)
        rows[name] = m
    winner = max(rows, key=lambda k: rows[k]["score"])
    return {"budget_tokens": budget, "strategies": rows, "winner": winner}


# ------------------------------------------------------------------- perf bench

def perf_bench(n_units, budget_ratio=0.3):
    units = make_corpus(n_units, seed=7)
    budget = int(sum(u["tokens"] for u in units) * budget_ratio)
    t0 = time.perf_counter()
    run_bench(units, budget)
    return time.perf_counter() - t0


# -------------------------------------------------------------------- CLI

def _cmd_bench(args):
    units = load_units(args.input) if args.input else make_corpus(args.synthetic, seed=args.seed)
    total = sum(u["tokens"] for u in units)
    budget = args.budget if args.budget is not None else max(1, int(total * 0.4))
    if budget < 0:
        raise BenchError("budget must be >= 0")
    report = run_bench(units, budget)
    report["units"] = len(units)
    report["total_tokens"] = total
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print("units=%d total_tokens=%d budget=%d" % (len(units), total, budget))
        print("%-16s %8s %10s %10s %8s" % ("strategy", "score", "weighted", "critical", "tokens"))
        for name in STRATEGIES:
            m = report["strategies"].get(name)
            if m:
                print("%-16s %8.4f %10.4f %10.4f %8d" % (
                    name, m["score"], m["weighted_retention"],
                    m["critical_retention"], m["kept_tokens"]))
        if report["winner"]:
            print("winner: %s" % report["winner"])
    return 0


def _cmd_perf(args):
    elapsed = perf_bench(args.units, args.ratio)
    print("perf: %d units in %.4fs" % (args.units, elapsed))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="context-truncation-bench",
        description="Benchmark context truncation strategies (head/tail/middle/hybrid) "
                    "on token, weighted and critical retention metrics.")
    sub = parser.add_subparsers(dest="cmd")
    p_bench = sub.add_parser("bench", help="run all strategies over a corpus")
    p_bench.add_argument("--input", help="JSON list of {id,kind,text,critical} units")
    p_bench.add_argument("--synthetic", type=int, default=500,
                         help="generate N synthetic units instead of --input")
    p_bench.add_argument("--budget", type=int, default=None,
                         help="token budget (default 40%% of corpus)")
    p_bench.add_argument("--seed", type=int, default=42)
    p_bench.add_argument("--json", action="store_true")
    p_perf = sub.add_parser("perf", help="timed benchmark for Performance section")
    p_perf.add_argument("--units", type=int, default=20000)
    p_perf.add_argument("--ratio", type=float, default=0.3)
    args = parser.parse_args(argv)
    if args.cmd not in ("bench", "perf"):
        parser.print_help()
        return 2
    try:
        if args.cmd == "bench":
            return _cmd_bench(args)
        return _cmd_perf(args)
    except (BenchError, OSError, json.JSONDecodeError) as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
