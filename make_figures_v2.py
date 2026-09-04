"""Figures for Run 012 (cross-fit, budget-matched, population-controlled).

Reads the committed artifact outputs/expansion/cross_fit.json (from
cross_fit.py). The commitment curve is the gold-free behavioural measurement
(forced mid-trace answer == final answer) on the FULL 1,000-trace set, parsed
from the forced-answer completions (the forcing template puts \\boxed{ in the
prompt, so the completion is the letter itself).

Outputs PNGs to outputs/expansion/figures/ and a figure_data.json audit dump.
No GPU, no network. Run: python make_figures_v2.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

HERE = os.path.dirname(os.path.abspath(__file__))
CF = os.path.join(HERE, "outputs", "expansion", "cross_fit.json")
FIGDIR = os.path.join(HERE, "outputs", "expansion", "figures")
os.makedirs(FIGDIR, exist_ok=True)

# Commitment: full 1,000-trace set, forced mid-trace answer == final answer.
COMMITMENT = {1: 0.530, 10: 0.570, 25: 0.676, 50: 0.856, 75: 0.947, 90: 0.971}

KPCT = ["k1", "k10", "k25", "k50", "k75", "k90"]
KX = [1, 10, 25, 50, 75, 90]
ABS = ["abs64", "abs128", "abs256", "abs512", "abs1024"]
AX = [64, 128, 256, 512, 1024]

PROBE = "#1f77b4"
TEXT = "#d62728"
LEN = "#7f7f7f"
CONF = "#2ca02c"


def load():
    with open(CF) as f:
        return json.load(f)


def fig1_delta(H):
    """Δ(probe − TF-IDF) vs cut, cross-fit, all cuts (every CI excludes 0)."""
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    ax.axhline(0, color="black", lw=1, ls="--", alpha=0.6)
    # k% cuts
    yk = [H[t]["delta_probe_minus_tfidf"] for t in KPCT]
    lok = [H[t]["delta_probe_minus_tfidf"] - H[t]["delta_ci"][0] for t in KPCT]
    hik = [H[t]["delta_ci"][1] - H[t]["delta_probe_minus_tfidf"] for t in KPCT]
    ax.errorbar(range(6), yk, yerr=[lok, hik], fmt="o-", color=PROBE, capsize=4,
                lw=2, label="k% cuts")
    # fixed-length cuts
    ya = [H[t]["delta_probe_minus_tfidf"] for t in ABS]
    loa = [H[t]["delta_probe_minus_tfidf"] - H[t]["delta_ci"][0] for t in ABS]
    hia = [H[t]["delta_ci"][1] - H[t]["delta_probe_minus_tfidf"] for t in ABS]
    ax.errorbar(range(6, 11), ya, yerr=[loa, hia], fmt="s-", color="#ff7f0e",
                capsize=4, lw=2, label="fixed-length cuts")
    labels = ["k1", "k10", "k25", "k50", "k75", "k90",
              "N64", "N128", "N256", "N512", "N1024"]
    ax.set_xticks(range(11))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Δ = AUC(probe) − AUC(TF-IDF)")
    ax.set_title("Budget-matched probe never beats text: Δ<0 at all 11 cuts,\n"
                 "every 95% CI excludes 0 (cross-fit, 213/196 negatives)")
    ax.legend(loc="lower left", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig1_delta.png"), dpi=200)
    plt.close(fig)


def fig2_readers(H):
    """Reader AUCs vs k, cross-fit, k% grid."""
    fig, ax = plt.subplots(figsize=(7, 4.3))
    ax.plot(KX, [H[t]["tfidf"] for t in KPCT], "o-", color=TEXT, lw=2, label="TF-IDF (text, 1 config)")
    ax.plot(KX, [H[t]["probe"] for t in KPCT], "s-", color=PROBE, lw=2, label="probe (activations, 1 layer)")
    ax.plot(KX, [H[t]["forced_conf"] for t in KPCT], "^-", color=CONF, lw=1.5, label="forced-confidence (gold-free)")
    ax.plot(KX, [H[t]["length_only"] for t in KPCT], "d--", color=LEN, lw=1.5, label="length-only (the leak)")
    ax.axhline(0.5, color="black", lw=1, ls=":", alpha=0.5)
    ax.set_xlabel("cut point k (% of thinking tokens)")
    ax.set_ylabel("cross-fit ROC-AUC")
    ax.set_title("Text leads the budget-matched probe at every cut (k% protocol)")
    ax.set_xticks(KX)
    ax.set_ylim(0.45, 0.88)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig2_readers.png"), dpi=200)
    plt.close(fig)


def fig3_decoupling(H):
    """THE figure: the answer commits (agreement -> ~1) long before its
    correctness becomes legible (best AUC stalls ~0.8). Two different quantities,
    both on [0.5, 1], plotted together to show the gap."""
    fig, ax = plt.subplots(figsize=(7.6, 4.7))
    xs = KX
    commit = [COMMITMENT[k] for k in xs]
    best = [H[t]["tfidf"] for t in KPCT]       # best correctness reader at each cut
    probe = [H[t]["probe"] for t in KPCT]
    # shade the legibility gap between commitment and the best correctness reader
    ax.fill_between(xs, best, commit, where=[c >= b for c, b in zip(commit, best)],
                    color="#9467bd", alpha=0.10, interpolate=True)
    ax.plot(xs, commit, "o-", color="#9467bd", lw=2.4,
            label="answer commitment  (forced = final answer)")
    ax.plot(xs, best, "s-", color=TEXT, lw=2.2,
            label="correctness legibility  (best reader, ROC-AUC)")
    ax.plot(xs, probe, "^--", color=PROBE, lw=1.4, alpha=0.8,
            label="— probe, for reference (ROC-AUC)")
    ax.axhline(0.5, color="black", lw=1, ls=":", alpha=0.5)
    ax.annotate("answer ~97% locked", (90, commit[-1]), textcoords="offset points",
                xytext=(-6, 8), ha="right", fontsize=8.5, color="#6a4a9c")
    ax.annotate("…but correctness only ~0.82 readable", (90, best[-1]),
                textcoords="offset points", xytext=(-6, -14), ha="right",
                fontsize=8.5, color="#a01f1f")
    ax.set_xlabel("cut point k (% of thinking tokens)")
    ax.set_ylabel("agreement  or  ROC-AUC   (both in [0.5, 1])")
    ax.set_title("Commitment precedes legibility: the model settles on its answer\n"
                 "long before an observer can tell whether it is right")
    ax.set_xticks(xs)
    ax.set_ylim(0.5, 1.02)
    ax.legend(loc="lower right", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig3_decoupling.png"), dpi=200)
    plt.close(fig)


def fig4_population_control(P):
    """Same 781 traces, both protocols: the leak is small and text is stable."""
    fig, ax = plt.subplots(figsize=(8, 4.6))
    xk, xa = range(6), range(6, 11)
    # TF-IDF: k% then fixed-length, on the same population
    ax.plot(xk, [P[t]["tfidf"] for t in KPCT], "o-", color=TEXT, lw=2, label="TF-IDF (text)")
    ax.plot(xa, [P[t]["tfidf"] for t in ABS], "o-", color=TEXT, lw=2)
    ax.plot(xk, [P[t]["probe"] for t in KPCT], "s-", color=PROBE, lw=2, label="probe")
    ax.plot(xa, [P[t]["probe"] for t in ABS], "s-", color=PROBE, lw=2)
    ax.plot(xk, [P[t]["length_only"] for t in KPCT], "d--", color=LEN, lw=1.5, label="length-only")
    ax.plot(xa, [P[t]["length_only"] for t in ABS], "d--", color=LEN, lw=1.5)
    ax.axvline(5.5, color="black", lw=1, ls=":", alpha=0.6)
    ax.axhline(0.5, color="black", lw=0.8, ls=":", alpha=0.4)
    ax.text(2.5, 0.46, "k% cuts (leak present)", ha="center", fontsize=9)
    ax.text(8, 0.46, "fixed-length (leak removed)", ha="center", fontsize=9)
    labels = ["k1", "k10", "k25", "k50", "k75", "k90",
              "N64", "N128", "N256", "N512", "N1024"]
    ax.set_xticks(range(11))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("cross-fit ROC-AUC")
    ax.set_ylim(0.44, 0.86)
    ax.set_title("Same 781 traces, both protocols: removing the leak drops\n"
                 "length-only to chance but barely moves text — the leak is real "
                 "but small,\nand does not explain text beating the probe")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig4_population_control.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    d = load()
    H, P = d["headline"], d["pop_control"]
    fig1_delta(H)
    fig2_readers(H)
    fig3_decoupling(H)
    fig4_population_control(P)
    audit = {
        "source": "outputs/expansion/cross_fit.json",
        "commitment_source": "full 1,000-trace set, gold-free forced-vs-final agreement",
        "commitment": COMMITMENT,
        "headline_delta": {t: {"d": H[t]["delta_probe_minus_tfidf"],
                               "ci": H[t]["delta_ci"],
                               "excl0": H[t]["delta_excludes_0"]} for t in KPCT + ABS},
    }
    with open(os.path.join(FIGDIR, "figure_data.json"), "w") as f:
        json.dump(audit, f, indent=1)
    print("wrote 4 figures + figure_data.json to", FIGDIR)


if __name__ == "__main__":
    main()
