# -*- coding: utf-8 -*-
"""self_test for context-truncation-bench — inherits scaffold-v2 defenses D1-D4."""
import contextlib
import io
import json
import os
import sys
import tempfile
import unittest

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", newline="\n")  # D3
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import context_truncation_bench as mod  # noqa: E402


def run_cli(argv):  # D1: catch SystemExit AND main() return code
    buf_out, buf_err = io.StringIO(), io.StringIO()
    code = 0
    try:
        with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
            rc = mod.main(argv)
            code = 0 if rc is None else rc
    except SystemExit as e:
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    except Exception as e:  # noqa: BLE001
        code = 1
        buf_err.write(repr(e))
    return code, buf_out.getvalue(), buf_err.getvalue()


CASES = []


def case(name, cat):
    def deco(fn):
        CASES.append((name, cat, fn))
        return fn
    return deco


TMP = tempfile.mkdtemp(prefix="ctb_selftest_")


def write_tmp(name, obj):
    p = os.path.join(TMP, name)
    with open(p, "w", encoding="utf-8", newline="") as fh:  # D3: newline=""
        if isinstance(obj, str):
            fh.write(obj)
        else:
            json.dump(obj, fh, ensure_ascii=False)
    return p


U = mod.make_unit

# ------------------------------------------------------------------ [A] functional

@case("A01_unit_token_count", "A")
def t():
    assert mod.token_count("hello  world\nfoo") == 3

@case("A02_make_unit_defaults", "A")
def t():
    u = U("x1", "fact", "a b")
    assert u["weight"] == 0.7 and u["tokens"] == 2 and u["critical"] is False

@case("A03_unknown_kind_default_weight", "A")
def t():
    assert mod.unit_weight(U("x", "lore", "t")) == 0.5

@case("A04_corpus_deterministic", "A")
def t():
    assert mod.make_corpus(20, seed=3)[4]["text"] == mod.make_corpus(20, seed=3)[4]["text"]

@case("A05_corpus_kinds_cycle", "A")
def t():
    ks = [u["kind"] for u in mod.make_corpus(10)]
    assert ks[0] == "system" and ks[1] == "instruction" and ks[4] == "noise"

@case("A06_corpus_critical_flags", "A")
def t():
    units = mod.make_corpus(70)
    crit = [u for u in units if u["critical"]]
    assert crit and all(u["kind"] == "instruction" for u in crit)

@case("A07_head_keeps_prefix", "A")
def t():
    units = [U("a", "fact", "one two"), U("b", "fact", "three"), U("c", "fact", "four five")]
    assert [u["id"] for u in mod.truncate_head(units, 3)] == ["a", "b"]

@case("A08_tail_keeps_suffix_order", "A")
def t():
    units = [U("a", "fact", "one two"), U("b", "fact", "three"), U("c", "fact", "four five")]
    assert [u["id"] for u in mod.truncate_tail(units, 3)] == ["b", "c"]

@case("A09_middle_keeps_center", "A")
def t():
    units = [U(str(i), "fact", "x " * (2 + i)) for i in range(5)]
    ids = [u["id"] for u in mod.truncate_middle(units, 12)]
    assert ids[0] != "0" and ids[-1] != "4" and ids

@case("A10_middle_noop_when_fits", "A")
def t():
    units = [U("a", "fact", "x"), U("b", "fact", "y")]
    assert [u["id"] for u in mod.truncate_middle(units, 100)] == ["a", "b"]

@case("A11_head_tail_respects_budget", "A")
def t():
    units = [U(str(i), "fact", "w" * 10) for i in range(10)]
    kept = mod.truncate_head_tail(units, 40)
    assert sum(u["tokens"] for u in kept) <= 40

@case("A12_top_importance_prefers_critical", "A")
def t():
    units = [U("n", "noise", "a b c d"), U("c", "instruction", "e f", critical=True)]
    kept = mod.truncate_top_importance(units, 3)
    assert [u["id"] for u in kept] == ["c"]

@case("A13_top_importance_restores_order", "A")
def t():
    units = [U("a", "fact", "x"), U("b", "system", "y"), U("c", "fact", "z")]
    assert [u["id"] for u in mod.truncate_top_importance(units, 10)] == ["a", "b", "c"]

@case("A14_hybrid_within_budget", "A")
def t():
    units = mod.make_corpus(60)
    kept = mod.truncate_hybrid(units, 100)
    assert sum(u["tokens"] for u in kept) <= 100 and kept

@case("A15_all_strategies_registered", "A")
def t():
    assert set(mod.STRATEGY_FNS) == set(mod.STRATEGIES)

@case("A16_metrics_shape", "A")
def t():
    units = mod.make_corpus(30)
    m = mod.retention_metrics(units, units[:5])
    assert set(m) == {"token_retention", "weighted_retention", "critical_retention", "kept_tokens"}

@case("A17_metrics_full_keeps_all", "A")
def t():
    units = mod.make_corpus(30)
    m = mod.retention_metrics(units, units)
    assert m["token_retention"] == 1.0 and m["critical_retention"] == 1.0

@case("A18_composite_score_bounds", "A")
def t():
    units = mod.make_corpus(30)
    s = mod.composite_score(mod.retention_metrics(units, units))
    assert 0.99 <= s <= 1.0001

@case("A19_bench_winner_exists", "A")
def t():
    r = mod.run_bench(mod.make_corpus(50), 60)
    assert r["winner"] in mod.STRATEGIES and len(r["strategies"]) == 6

@case("A20_bench_budget_from_cli", "A")
def t():
    p = write_tmp("units.json", [{"id": "a", "kind": "fact", "text": "x y z"}])
    code, out, _ = run_cli(["bench", "--input", p, "--budget", "2", "--json"])
    assert code == 0 and json.loads(out)["budget_tokens"] == 2

@case("A21_cli_perf_subcommand", "A")
def t():
    code, out, _ = run_cli(["perf", "--units", "200", "--ratio", "0.5"])
    assert code == 0 and out.startswith("perf: 200 units")

@case("A22_load_units_roundtrip", "A")
def t():
    p = write_tmp("u2.json", [{"id": "中文", "kind": "系统", "text": "你好 世界"}])
    units = mod.load_units(p)
    assert units[0]["id"] == "中文" and units[0]["kind"] == "系统"

@case("A23_weights_table_values", "A")
def t():
    assert mod.WEIGHTS["system"] > mod.WEIGHTS["fact"] > mod.WEIGHTS["noise"]

@case("A24_run_bench_empty_note", "A")
def t():
    r = mod.run_bench([], 10)
    assert r["winner"] is None and "empty" in r["note"]

# ------------------------------------------------------------------ [B] exception / CLI contract

@case("B01_cli_no_args_exit2", "B")
def t():
    code, _, _ = run_cli([])
    assert code == 2

@case("B02_cli_bench_missing_input_exit2", "B")
def t():
    code, _, err = run_cli(["bench", "--input", "nope_missing.json"])
    assert code == 2 and "error" in err

@case("B03_cli_unknown_subcommand_exit2", "B")
def t():
    code, _, _ = run_cli(["frobnicate"])
    assert code == 2

@case("B04_negative_budget_rejected", "B")
def t():
    code, _, err = run_cli(["bench", "--synthetic", "5", "--budget", "-1"])
    assert code == 2 and "budget" in err

@case("B05_load_units_not_list", "B")
def t():
    p = write_tmp("obj.json", {"not": "a list"})
    try:
        mod.load_units(p)
        assert False
    except mod.BenchError as e:
        assert "JSON list" in str(e)

@case("B06_make_unit_non_str_text", "B")
def t():
    try:
        U("x", "fact", 123)
        assert False
    except mod.BenchError:
        pass

@case("B07_synthetic_must_be_int", "B")
def t():
    code, _, _ = run_cli(["bench", "--synthetic", "abc"])
    assert code == 2

@case("B08_json_mode_valid_json", "B")
def t():
    code, out, _ = run_cli(["bench", "--synthetic", "10", "--budget", "20", "--json"])
    assert code == 0 and isinstance(json.loads(out), dict)

@case("B09_perf_bad_ratio_float", "B")
def t():
    code, _, _ = run_cli(["perf", "--units", "10", "--ratio", "zzz"])
    assert code == 2

@case("B10_main_returns_int_not_raise", "B")
def t():
    assert isinstance(mod.main(["bench", "--synthetic", "3", "--budget", "5"]), int)

@case("B11_strategies_keys_match_fns", "B")
def t():
    for s in mod.STRATEGIES:
        assert callable(mod.STRATEGY_FNS[s])

@case("B12_budget_zero_returns_empty", "B")
def t():
    units = [U("a", "fact", "x")]
    assert mod.truncate_head(units, 0) == []

@case("B13_run_cli_captures_stderr_only", "B")
def t():
    code, out, err = run_cli(["bench", "--input", "ghost.json"])
    assert out == "" and err != ""

@case("B14_metrics_zero_total_no_div0", "B")
def t():
    m = mod.retention_metrics([], [])
    assert m["token_retention"] == 0.0

@case("B15_composite_zero_case", "B")
def t():
    assert mod.composite_score({"weighted_retention": 0, "critical_retention": 0}) == 0.0

@case("B16_cli_help_text_mentions_strategies", "B")
def t():
    code, out, _ = run_cli(["--help"])
    assert code == 0 and "truncation" in out

@case("B17_bench_json_units_field", "B")
def t():
    code, out, _ = run_cli(["bench", "--synthetic", "12", "--budget", "30", "--json"])
    assert json.loads(out)["units"] == 12

@case("B18_seed_changes_corpus", "B")
def t():
    a = mod.make_corpus(10, seed=1)
    b = mod.make_corpus(10, seed=2)
    assert a[0]["text"] != b[0]["text"]

# ------------------------------------------------------------------ [C] extreme edge

@case("C01_empty_corpus_cli", "C")
def t():
    code, out, _ = run_cli(["bench", "--synthetic", "0", "--budget", "5", "--json"])
    assert code == 0 and json.loads(out)["winner"] is None

@case("C02_budget_exceeds_total", "C")
def t():
    units = mod.make_corpus(10)
    kept = mod.truncate_head(units, 10 ** 9)
    assert len(kept) == 10

@case("C03_single_unit_over_budget", "C")
def t():
    units = [U("big", "fact", "x " * 100)]
    for fn in mod.STRATEGY_FNS.values():
        assert sum(u["tokens"] for u in fn(units, 5)) <= 5

@case("C04_single_unit_under_budget", "C")
def t():
    units = [U("s", "fact", "x")]
    assert mod.truncate_head(units, 5)[0]["id"] == "s"

@case("C05_zero_token_units", "C")
def t():
    units = [U("e1", "fact", ""), U("e2", "fact", "")]
    assert len(mod.truncate_head(units, 1)) == 2

@case("C06_unicode_text_tokens", "C")
def t():
    u = U("u", "fact", "你好世界 こんにちは Привет мир")
    assert u["tokens"] == 4

@case("C07_emoji_corpus_survives", "C")
def t():
    units = [U("e", "fact", "🚀🔥 x")]
    kept = mod.truncate_hybrid(units, 10)
    assert kept and kept[0]["text"] == "🚀🔥 x"

@case("C08_crlf_text_tokenized", "C")
def t():
    assert mod.token_count("a\r\nb\rc") == 3

@case("C09_all_critical_kept_first", "C")
def t():
    units = [U(str(i), "instruction", "w", critical=True) for i in range(5)]
    kept = mod.truncate_top_importance(units, 3)
    assert len(kept) == 3 and all(u["critical"] for u in kept)

@case("C10_no_critical_metrics_returns_1", "C")
def t():
    units = [U("a", "noise", "x")]
    assert mod.retention_metrics(units, units[:0] or [])["critical_retention"] == 1.0

@case("C11_huge_text_unit", "C")
def t():
    u = U("h", "fact", "word " * 100000)
    assert u["tokens"] == 100000
    assert mod.truncate_head([u], 50) == []

@case("C12_deep_many_units_strategy_speed", "C")
def t():
    units = mod.make_corpus(5000, seed=9)
    for fn in mod.STRATEGY_FNS.values():
        assert fn(units, 1000) is not None

@case("C13_negative_seed_ok", "C")
def t():
    assert mod.make_corpus(5, seed=-7)

@case("C14_unicode_kind_id_json_file", "C")
def t():
    p = write_tmp("uni.json", [{"id": "модуль", "kind": "事実", "text": "日本語 テスト"}])
    code, out, _ = run_cli(["bench", "--input", p, "--budget", "1", "--json"])
    assert code == 0 and json.loads(out)["units"] == 1

@case("C15_middle_with_giant_edge_units", "C")
def t():
    units = [U("l", "fact", "x " * 100), U("m", "fact", "y"), U("r", "fact", "z " * 100)]
    kept = mod.truncate_middle(units, 3)
    assert [u["id"] for u in kept] == ["m"]

@case("C16_head_tail_single_side", "C")
def t():
    units = [U("a", "fact", "one"), U("b", "fact", "two")]
    kept = mod.truncate_head_tail(units, 2)
    assert kept and sum(u["tokens"] for u in kept) <= 2

@case("C17_metrics_partial_critical", "C")
def t():
    units = [U("a", "instruction", "x", critical=True), U("b", "noise", "y")]
    m = mod.retention_metrics(units, [units[1]])
    assert m["critical_retention"] == 0.0

@case("C18_corrupt_input_file_exit2", "C")
def t():
    p = write_tmp("bad.json", "{not json at all")
    code, _, _ = run_cli(["bench", "--input", p])
    assert code == 2

@case("C19_benchmark_synthetic_1_unit", "C")
def t():
    code, out, _ = run_cli(["bench", "--synthetic", "1", "--budget", "1000", "--json"])
    assert code == 0 and json.loads(out)["winner"]

@case("C20_hybrid_head_tail_disjoint", "C")
def t():
    units = mod.make_corpus(80)
    kept = mod.truncate_hybrid(units, 150)
    ids = [id(u) for u in kept]
    assert len(ids) == len(set(ids))


class TestAll(unittest.TestCase):
    pass


def _build():
    for i, (name, cat, fn) in enumerate(CASES):
        def make(f):
            def m(self):
                f()
            return m
        slug = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in name)
        setattr(TestAll, "test_%s_%03d_%s" % (cat, i, slug), make(fn))
_build()


class TestMeta(unittest.TestCase):
    def test_case_count_ge_60(self):
        self.assertGreaterEqual(len(CASES), 60)

    def test_categories_present(self):
        self.assertEqual({c for _, c, _ in CASES}, {"A", "B", "C"})


if __name__ == "__main__":
    r = unittest.TextTestRunner(verbosity=0).run(
        unittest.defaultTestLoader.loadTestsFromModule(sys.modules["__main__"]))
    total, fails = r.testsRun, len(r.failures) + len(r.errors)
    print("SUMMARY: PASS %d/%d" % (total - fails, total))  # D4
    sys.exit(0 if fails == 0 else 1)
