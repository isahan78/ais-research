import sys, os
sys.path.insert(0, "/Users/IsahanKhan/ais-research")
from experiment import analysis
OUT = os.path.dirname(os.path.abspath(__file__))
results = {"per_k": {str(k): {"n_test": 60, "best_layer": "layer_18",
            "per_layer": {"layer_18": {"auc": a, "auc_ci95": [a-.08, a+.08]}}}
            for k, a in [(10,.61),(25,.68),(50,.74),(75,.80),(90,.86)]}}
def pt(a): return {"auc": a, "auc_ci95": [a-.09, a+.09],
                   "test_row_keys": None, "test_labels": None, "test_scores": None}
baselines = {
  "forced_answer":  {10:pt(.55),25:pt(.63),50:pt(.72),75:pt(.79),90:pt(.85)},
  "llm_judge":      {10:pt(.58),25:pt(.64),50:pt(.69),75:pt(.75),90:pt(.82)},
  "text_classifier":{10:pt(.53),25:pt(.58),50:pt(.65),75:pt(.71),90:pt(.78)},
  "crude_floor":    {10:pt(.52),25:pt(.53),50:pt(.54),75:pt(.55),90:pt(.56)},
}
out = analysis.analyze(results, baselines, n_bootstrap=50)
print(analysis.figure_delta_curve(out, os.path.join(OUT,"multi_fig1.png")))
print(analysis.figure_baseline_comparison(out, os.path.join(OUT,"multi_fig2.png")))
# missing-baseline variant
out2 = analysis.analyze(results, {"forced_answer": baselines["forced_answer"],
                                  "llm_judge": {}, "text_classifier": {}, "crude_floor": {}},
                        n_bootstrap=50)
print(analysis.figure_baseline_comparison(out2, os.path.join(OUT,"multi_fig2_missing.png")))
