# Red-team evidence (2026-08-30)

The scripts two adversarial agents used to attack this project's own headline
result, preserved verbatim. Their findings are documented in RESULTS.md Run 007;
these are the receipts. The most important one is `b_forced.py`, which proved
the forced-answer baseline was scored against the gold answer (AUC 0.000 on
rows where it was not simply copying the trace's final answer) — the finding
that withdrew our own headline four days before submission.

Also here: the split/label/truncation/alignment audit scripts that verified
the measurement pipeline itself was clean (`check_*.py`), and the probe-tuning
audit (`e_probe.py`) showing the reported non-monotonicity was an artifact of
an unswept C and test-side layer selection.
