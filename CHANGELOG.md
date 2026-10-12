# Changelog

All notable changes to this project are documented here.
Format based on Keep a Changelog; versioning: SemVer.

## [0.1.1] - 2026-10-13

### Fixed / Revised (round-2 iteration)
- fixed CLI error contract: main() now catches OSError/JSONDecodeError -> exit 2 (was uncaught exit 1)
- truncate_middle rewritten as centred sliding-window search after budget-violation and head-bias bugs found by C-cases (C03/C15/A09)
- case count raised into the 60-85 band; all suites re-run green

### Verification
- self_test re-run green after revisions: SUMMARY: PASS 64/64 (62 cases + 2 meta; >=60 gate met)
- CLI error contract re-verified (exit 0/1/2); four scaffold-v2 defenses inherited.

## [0.1.0] - 2026-10-13

### Added
- Core module `context_truncation_bench.py`: bench (all six strategies over a corpus) / perf (timed benchmark), pure standard library.
- argparse CLI with `--json` machine output and 0/1/2 exit-code contract.
- `self_test.py` inheriting scaffold-v2 defenses: SystemExit capture, unicode-safe
  regexes, `newline=""` fixtures, and the `SUMMARY: PASS n/n` last-line contract.
- README (Performance section with [Measured] numbers and in-repo repro commands),
  MIT LICENSE, ALIGNMENT.md (16-item rubric).
