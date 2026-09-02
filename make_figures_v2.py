"""Figures for the final (Run 011) expanded dataset.

Every number is read from the committed artifact
outputs/expansion/final_table.json (produced by compute_final_table.py), except
the commitment curve, which is the original-dataset behavioral measurement
(forced mid-trace answer == final trace answer) stated in RESULTS.md; the
expanded forced-answer generator hit a lower parse rate, so the commitment
curve is reported on the original dataset.

Outputs PNGs to outputs/expansion/figures/ and a figure_data.json audit dump.
No GPU, no network. Run: python make_figures_v2.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
TABLE = os.path.join(HERE, "outputs", "expansion", "final_table.json")
FIGDIR = os.path.join(HERE, "outputs", "expansion", "figures")
os.makedirs(FIGDIR, exist_ok=True)

# Commitment curve: original dataset, forced mid-trace answer == final answer.
COMMITMENT = {10: 0.595, 25: 0.685, 50: 0.865, 75: 0.927, 90: 0.965}

KPCT = ["k1", "k10", "k25", "k50", "k75", "k90"]
KX = [1, 10, 25, 50, 75, 90]
ABS = ["abs64", "abs128", "abs256", "abs512", "abs1024"]
AX = [64, 128, 256, 512, 1024]

PROBE = "#1f77b4"
TEXT = "#d62728"
LEN = "#7f7f7f"
CONF = "#2ca02c"


def load():
    with open(TABLE) as f:
        return json.load(f)


def fig1_delta(d):
    """Δ(probe − best text) vs k, with paired-bootstrap CIs, k% grid."""
    fig, ax = plt.subplots(figsize=(7, 4.3))
    ys = [d[k]["delta"] for k in KPCT]
    lo = [d[k]["delta"] - d[k]["delta_ci"][0] for k in KPCT]
    hi = [d[k]["delta_ci"][1] - d[k]["delta"] for k in KPCT]
    ax.axhline(0, color="black", lw=1, ls="--", alpha=0.6)
    ax.errorbar(KX, ys, yerr=[lo, hi], fmt="o-", color=PROBE, capsize=4, lw=2)
    for k, x, y in zip(KPCT, KX, ys):
        if d[k]["delta_excludes_0"]:
            ax.annotate("*", (x, y), textcoords="offset points",
                        xytext=(0, 9), ha="center", fontsize=15, color=PROBE)
    ax.set_xlabel("cut point k (% of thinking tokens)")
    ax.set_ylabel("Δ = AUC(probe) − AUC(best text reader)")
    ax.set_title("Probe never beats text: Δ negative at every cut\n"
                 "(* = 95% paired-bootstrap CI excludes 0; n_test=337, 77 neg)")
    ax.set_xticks(KX)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig1_delta.png"), dpi=200)
    plt.close(fig)


def fig2_readers(d):
    """Reader AUCs vs k, k% grid."""
    fig, ax = plt.subplots(figsize=(7, 4.3))
    ax.plot(KX, [d[k]["tfidf"] for k in KPCT], "o-", color=TEXT, lw=2, label="tuned TF-IDF (text)")
    ax.plot(KX, [d[k]["probe"] for k in KPCT], "s-", color=PROBE, lw=2, label="linear probe (activations)")
    ax.plot(KX, [d[k]["forced_conf"] for k in KPCT], "^-", color=CONF, lw=1.5, label="forced-confidence (gold-free)")
    ax.plot(KX, [d[k]["length_only"] for k in KPCT], "d--", color=LEN, lw=1.5, label="length-only (the leak)")
    ax.axhline(0.5, color="black", lw=1, ls=":", alpha=0.5)
    ax.set_xlabel("cut point k (% of thinking tokens)")
    ax.set_ylabel("held-out ROC-AUC")
    ax.set_title("Text reader leads the probe at every cut (k% protocol)")
    ax.set_xticks(KX)
    ax.set_ylim(0.45, 0.9)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig2_readers.png"), dpi=200)
    plt.close(fig)


def fig3_commitment():
    """Forced mid-trace answer agreement with final answer (original dataset)."""
    fig, ax = plt.subplots(figsize=(7, 4.3))
    xs = sorted(COMMITMENT)
    ax.plot(xs, [COMMITMENT[k] for k in xs], "o-", color="#9467bd", lw=2)
    ax.axhline(1.0, color="black", lw=1, ls=":", alpha=0.5)
    for k in xs:
        ax.annotate(f"{COMMITMENT[k]:.0%}", (k, COMMITMENT[k]),
                    textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
    ax.set_xlabel("cut point k (% of thinking tokens)")
    ax.set_ylabel("agreement: forced answer = final answer")
    ax.set_title("The model commits early: by halfway the answer is usually set\n"
                 "(gold-free, original dataset)")
    ax.set_xticks(xs)
    ax.set_ylim(0.5, 1.02)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig3_commitment.png"), dpi=200)
    plt.close(fig)


def fig4_protocol(d):
    """k% vs fixed-length: probe and TF-IDF, side by side."""
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3), sharey=True)
    # left: k% protocol (has the length leak)
    a1.plot(KX, [d[k]["tfidf"] for k in KPCT], "o-", color=TEXT, lw=2, label="TF-IDF")
    a1.plot(KX, [d[k]["probe"] for k in KPCT], "s-", color=PROBE, lw=2, label="probe")
    a1.plot(KX, [d[k]["length_only"] for k in KPCT], "d--", color=LEN, lw=1.5, label="length-only")
    a1.set_title("k% cuts (length leaks in)")
    a1.set_xlabel("k (% of thinking)")
    a1.set_ylabel("held-out ROC-AUC")
    a1.set_xticks(KX)
    a1.legend(loc="lower right", fontsize=9)
    # right: fixed-length protocol (leak removed by construction)
    a2.plot(AX, [d[k]["tfidf"] for k in ABS], "o-", color=TEXT, lw=2, label="TF-IDF")
    a2.plot(AX, [d[k]["probe"] for k in ABS], "s-", color=PROBE, lw=2, label="probe")
    a2.plot(AX, [d[k]["length_only"] for k in ABS], "d--", color=LEN, lw=1.5, label="length-only")
    a2.set_title("fixed-length cuts (leak removed)")
    a2.set_xlabel("N (thinking tokens kept)")
    a2.set_xscale("log", base=2)
    a2.set_xticks(AX)
    a2.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    a2.legend(loc="lower right", fontsize=9)
    for a in (a1, a2):
        a.axhline(0.5, color="black", lw=1, ls=":", alpha=0.5)
        a.set_ylim(0.45, 0.9)
    fig.suptitle("Removing the length leak narrows text's lead to a near-tie — "
                 "the probe still never wins", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(os.path.join(FIGDIR, "fig4_protocol_comparison.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    d = load()
    fig1_delta(d)
    fig2_readers(d)
    fig3_commitment()
    fig4_protocol(d)
    audit = {
        "source": "outputs/expansion/final_table.json",
        "commitment_source": "original dataset (RESULTS.md), gold-free agreement",
        "commitment": COMMITMENT,
        "kpct": {k: {kk: d[k][kk] for kk in ("probe", "tfidf", "length_only",
                 "forced_conf", "delta", "delta_ci", "delta_excludes_0")} for k in KPCT},
        "abs": {k: {kk: d[k][kk] for kk in ("probe", "tfidf", "length_only",
                "delta", "delta_ci", "delta_excludes_0")} for k in ABS},
    }
    with open(os.path.join(FIGDIR, "figure_data.json"), "w") as f:
        json.dump(audit, f, indent=1)
    print("wrote 4 figures + figure_data.json to", FIGDIR)


if __name__ == "__main__":
    main()
