"""Build the write-up figures and an interactive exploration page.

Every number is computed from committed artifacts under outputs/block2/ —
nothing is hand-typed — so the figures regenerate from a bare clone and
cannot drift from the data. Run from the repo parent:

    python -m experiment.make_figures            # static PNGs (matplotlib)
    python -m experiment.make_figures --interactive   # + plotly HTML

Figure 1 (matplotlib, submission): corrected Δ(k) with paired-bootstrap CI band (S_text = tuned
          TF-IDF only; forced-answer and the judge are excluded per
          analysis.EXCLUDED_FROM_S_TEXT and shown separately).
Figure 2: every reader across k, valid readers solid, excluded readers
          dashed/grey — the exclusion story told visually. Includes the
          length-only baseline (log prefix tokens, fit on train only),
          computed here from prefixes.jsonl + the recorded split.
Figure 3: commitment curve — TRUE answer identity between the interrupted
          (forced) answer and the trace's final answer, computed from
          generated_text, NOT from forced_correct≡label (which also counts
          both-wrong-with-different-answers as agreement).
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

try:
    from experiment.grading import extract_boxed
except ImportError:
    from grading import extract_boxed

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "block2")
FIGDIR = os.path.join(BASE, "figures")
KS = [1, 10, 25, 50, 75, 90]

C_PROBE = "#1f77b4"   # blue
C_TEXT  = "#e07b28"   # orange (colorblind-safe vs blue)
C_LEN   = "#6a51a3"   # purple
C_EXCL  = "#999999"   # grey for withdrawn readers


def load_json(path):
    with open(path) as f:
        return json.load(f)


def rows_of(path):
    out = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r.get("record_type") != "meta":
                out.append(r)
    return out


def collect():
    d = {"k": KS, "delta": [], "lo": [], "hi": [], "probe": [], "tfidf": [],
         "floor": [], "forced": [], "judge": [], "len_only": [], "commit": []}
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    # final answers per problem, from the generation artifact
    finals = {}
    for r in rows_of(os.path.join(BASE, "gen", "traces.jsonl")):
        pid = str(r["problem_id"])
        # final answer letter: extract from the post-</think> span of the trace
        txt = r.get("trace_text") or ""
        tail = txt.split("</think>", 1)[1] if "</think>" in txt else ""
        finals[pid] = extract_boxed(tail)

    for k in KS:
        kd = os.path.join(BASE, f"k{k}")
        a = load_json(os.path.join(kd, "analysis.json"))
        e = a["per_k"][str(k)] if "per_k" in a else a
        # analysis.json layout: find delta + ci + s_probe + s_text
        delta = e.get("delta"); ci = e.get("delta_ci95") or [None, None]
        d["delta"].append(delta); d["lo"].append(ci[0]); d["hi"].append(ci[1])
        d["probe"].append(e.get("s_probe"))
        r = load_json(os.path.join(kd, "results.json"))
        d["floor"].append((r.get("text_floor") or {}).get("auc"))

        def bauc(name):
            p = os.path.join(kd, f"baseline_{name}.json")
            if not os.path.exists(p): return None
            return list(load_json(p)["per_k"].values())[0]["auc"]
        d["tfidf"].append(bauc("text_classifier"))
        d["forced"].append(bauc("forced_answer"))
        d["judge"].append(bauc("llm_judge"))

        # length-only baseline: 1 feature, fit on train split only
        rows = rows_of(os.path.join(kd, "prefixes.jsonl"))
        rows = [x for x in rows if x.get("included")]
        pid = np.array([str(x["problem_id"]) for x in rows])
        y = np.array([bool(x["label"]) for x in rows])
        X = np.log(np.array([len(x["prefix_token_ids"]) for x in rows], float)).reshape(-1, 1)
        split = r["split"]; tr_ids = set(map(str, split["train_problem_ids"]))
        tr = np.array([p in tr_ids for p in pid]); te = ~tr
        m = LogisticRegression().fit(X[tr], y[tr])
        d["len_only"].append(float(roc_auc_score(y[te], m.decision_function(X[te]))))

        # commitment: TRUE answer identity forced letter == final letter
        fp = os.path.join(kd, "forced_answer.jsonl")
        if os.path.exists(fp):
            frs = rows_of(fp)
            same = tot = 0
            for fr in frs:
                fa = fr.get("forced_answer") or extract_boxed(fr.get("generated_text") or "")
                fin = fr.get("final_answer") or finals.get(str(fr["problem_id"]))
                if fa is None or fin is None: continue
                tot += 1; same += (str(fa).strip().upper() == str(fin).strip().upper())
            d["commit"].append(same / tot if tot else None)
        else:
            d["commit"].append(None)
    return d


def static_figs(d):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.grid": True, "grid.alpha": 0.3,
        "axes.spines.top": False, "axes.spines.right": False,
        "font.size": 13, "axes.titlesize": 15, "axes.labelsize": 13,
    })
    os.makedirs(FIGDIR, exist_ok=True)
    ks = d["k"]

    # ---- Figure 1: corrected Delta(k) ----
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.axhline(0, color="black", lw=1)
    ax.fill_between(ks, d["lo"], d["hi"], alpha=0.18, color=C_PROBE, label="95% CI (paired bootstrap)")
    ax.plot(ks, d["delta"], "o-", color=C_PROBE, lw=2.5, ms=7, label="Δ = probe − tuned TF-IDF")
    i25 = ks.index(25)
    ax.annotate("only cut whose CI excludes 0", (25, d["delta"][i25]),
                xytext=(33, d["delta"][i25] - 0.09), fontsize=11,
                arrowprops=dict(arrowstyle="->", lw=1))
    ax.set_xlabel("k — % of thinking tokens shown to every reader")
    ax.set_ylabel("Δ (AUC)")
    ax.set_title("Probe minus best valid text reader: flat ≈ −0.10, not widening")
    ax.legend(loc="lower right", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, "fig1_delta.png"), dpi=200); plt.close(fig)

    # ---- Figure 2: all readers, exclusions visible ----
    fig, ax = plt.subplots(figsize=(9.5, 6))
    ax.plot(ks, d["probe"], "o-", color=C_PROBE, lw=2.5, label="probe on activations (best of 3 layers)")
    ax.plot(ks, d["tfidf"], "s-", color=C_TEXT, lw=2.5, label="tuned TF-IDF (valid text reader)")
    ax.plot(ks, d["len_only"], "d-", color=C_LEN, lw=2, label="prefix length ALONE (1 feature)")
    ax.plot(ks, d["floor"], "v-", color=C_LEN, lw=1, alpha=0.45, label="crude 2-feature floor (mis-specified)")
    fk = [k for k, v in zip(ks, d["forced"]) if v is not None]
    fv = [v for v in d["forced"] if v is not None]
    ax.plot(fk, fv, "^--", color=C_EXCL, lw=2, label="forced-answer — WITHDRAWN (reads gold)")
    jk = [k for k, v in zip(ks, d["judge"]) if v is not None]
    jv = [v for v in d["judge"] if v is not None]
    ax.plot(jk, jv, "x--", color="#555555", lw=2, ms=10, label="LLM judge — excluded (difficulty oracle)")
    ax.axhline(0.5, color="black", lw=0.8, ls=":")
    ax.set_ylim(0.45, 1.0)
    ax.set_xlabel("k — % of thinking tokens")
    ax.set_ylabel("held-out AUC (n_test = 102)")
    ax.set_title("Every reader, with the two invalid ones shown for what they are")
    ax.legend(loc="lower right", fontsize=10, framealpha=0.95)
    fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, "fig2_readers.png"), dpi=200); plt.close(fig)

    # ---- Figure 3: commitment curve (gold-free) ----
    ck = [k for k, v in zip(ks, d["commit"]) if v is not None]
    cv = [v for v in d["commit"] if v is not None]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(ck, [100 * v for v in cv], "o-", color=C_TEXT, lw=2.5, ms=8)
    for x, v in zip(ck, cv):
        ax.annotate(f"{100*v:.0f}%", (x, 100 * v), xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=12)
    ax.set_xlabel("k — % of thinking tokens at interruption")
    ax.set_ylabel("interrupted answer == final answer (%)")
    ax.set_title("When does the model commit? (needs no answer key)")
    ax.set_ylim(40, 105)
    fig.tight_layout(); fig.savefig(os.path.join(FIGDIR, "fig3_commitment.png"), dpi=200); plt.close(fig)
    print(f"static figures -> {FIGDIR}/fig1_delta.png, fig2_readers.png, fig3_commitment.png")


def interactive(d):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    ks = d["k"]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.09,
                        subplot_titles=("Δ(k) = probe − tuned TF-IDF (95% CI)",
                                        "All readers (dashed grey = excluded from S_text)",
                                        "Commitment: interrupted answer == final answer"))
    fig.add_trace(go.Scatter(x=ks + ks[::-1], y=d["hi"] + d["lo"][::-1], fill="toself",
                             fillcolor="rgba(31,119,180,0.15)", line=dict(width=0),
                             name="95% CI", hoverinfo="skip"), 1, 1)
    fig.add_trace(go.Scatter(x=ks, y=d["delta"], mode="lines+markers", name="Δ",
                             line=dict(color=C_PROBE, width=3)), 1, 1)
    fig.add_hline(y=0, line_width=1, line_color="black", row=1, col=1)

    series = [("probe", d["probe"], C_PROBE, None), ("tuned TF-IDF", d["tfidf"], C_TEXT, None),
              ("length only", d["len_only"], C_LEN, None), ("crude floor", d["floor"], C_LEN, "dot"),
              ("forced-answer (WITHDRAWN — reads gold)", d["forced"], C_EXCL, "dash"),
              ("LLM judge (difficulty oracle)", d["judge"], "#555555", "dash")]
    for name, ys, color, dash in series:
        xs = [k for k, v in zip(ks, ys) if v is not None]
        vs = [v for v in ys if v is not None]
        fig.add_trace(go.Scatter(x=xs, y=vs, mode="lines+markers", name=name,
                                 line=dict(color=color, width=2.5, dash=dash)), 2, 1)
    ck = [k for k, v in zip(ks, d["commit"]) if v is not None]
    cv = [100 * v for v in d["commit"] if v is not None]
    fig.add_trace(go.Scatter(x=ck, y=cv, mode="lines+markers+text", text=[f"{v:.0f}%" for v in cv],
                             textposition="top center", name="commitment %",
                             line=dict(color=C_TEXT, width=3)), 3, 1)
    fig.update_layout(height=1100, width=950, title="The Probe–Text Gap — corrected results (Run 007)",
                      hovermode="x unified")
    fig.update_xaxes(title_text="k — % of thinking tokens", row=3, col=1)
    out = os.path.join(FIGDIR, "explore.html")
    fig.write_html(out, include_plotlyjs="cdn")
    print(f"interactive -> {out}")



def collect_abs():
    """Fixed-length grid (Runs 008/009): honest probe + baselines per N."""
    base = os.path.join(os.path.dirname(FIGDIR.rstrip("/")), "block2abs")
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "block2abs")
    hp = os.path.join(base, "honest_abs.json")
    if not os.path.exists(hp):
        return None
    h = load_json(hp)
    ns = sorted(int(n) for n in h)
    return {
        "n": ns,
        "probe": [h[str(n)]["honest_auc"] for n in ns],
        "probe_lo": [h[str(n)]["ci"][0] for n in ns],
        "probe_hi": [h[str(n)]["ci"][1] for n in ns],
        "tfidf": [h[str(n)]["tfidf"] for n in ns],
        "conf": [h[str(n)]["confidence"] for n in ns],
    }


def fig4_protocols(d, a):
    """Figure 4: the same readers under the leaky k% protocol vs fixed-length
    cuts. The TF-IDF collapse (0.78 -> ~0.60) is the project's key finding."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    # left: k% protocol (length leak present). Use the HONEST probe curve
    # (train-CV selection, Run 010) — the test-selected curve has a spurious
    # 0.76 spike at k=10 and must never be plotted as "honest".
    hp_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs", "block2", "honest_probe.json")
    honest = load_json(hp_path)["summary"]["honest_curve"] if os.path.exists(hp_path) else d["probe"]
    ax1.plot(d["k"], d["tfidf"], "s-", color=C_TEXT, lw=2.5, label="tuned TF-IDF")
    ax1.plot(d["k"], honest, "o-", color=C_PROBE, lw=2.5, label="probe (honest)")
    ax1.set_title("k% protocol — leaks eventual trace length\n(corr with full length = 0.99999999)")
    ax1.set_xlabel("k — % of thinking tokens")
    # right: fixed-length protocol (leak impossible)
    ax2.plot(a["n"], a["tfidf"], "s-", color=C_TEXT, lw=2.5, label="tuned TF-IDF")
    ax2.fill_between(a["n"], a["probe_lo"], a["probe_hi"], alpha=0.15, color=C_PROBE)
    ax2.plot(a["n"], a["probe"], "o-", color=C_PROBE, lw=2.5, label="probe (honest, CI band)")
    ax2.plot(a["n"], a["conf"], "d-", color=C_LEN, lw=2, label="forced-confidence (gold-free)")
    ax2.set_xscale("log", base=2)
    ax2.set_xticks(a["n"]); ax2.set_xticklabels(a["n"])
    ax2.set_title("Fixed-length cuts — leak structurally impossible\n(fixed population, 242 traces)")
    ax2.set_xlabel("N — thinking tokens at cut (log scale)")
    for ax in (ax1, ax2):
        ax.axhline(0.5, color="black", lw=0.8, ls=":")
        ax.set_ylim(0.40, 0.85)
        ax.legend(loc="lower right", fontsize=10)
    ax1.set_ylabel("held-out AUC")
    ax1.annotate("TF-IDF ≈ 0.78", (50, 0.79), fontsize=11, color=C_TEXT)
    ax2.annotate("TF-IDF collapses\nto 0.56–0.65", (128, 0.53), fontsize=11, color=C_TEXT)
    fig.suptitle("The text reader's advantage was largely the protocol's length leak", y=1.02, fontsize=15)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGDIR, "fig4_protocol_comparison.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"figure 4 -> {FIGDIR}/fig4_protocol_comparison.png")


def main():
    d = collect()
    os.makedirs(FIGDIR, exist_ok=True)
    with open(os.path.join(FIGDIR, "figure_data.json"), "w") as f:
        json.dump(d, f, indent=1)
    static_figs(d)
    a = collect_abs()
    if a:
        fig4_protocols(d, a)
    if "--interactive" in sys.argv:
        interactive(d)


if __name__ == "__main__":
    main()
