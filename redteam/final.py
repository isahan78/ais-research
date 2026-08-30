import json
from experiment.config import CONFIG
r = json.load(open(CONFIG.results_path))
for layer, d in r["per_layer"].items():
    print(f"  {layer}: AUC={d['auc']} CI95={d['auc_ci95']}")
fl = r["shuffled_floor"]
tf = r["text_floor"]
best = r["per_layer"][r["best_layer"]]
print(f"  shuffled floor (max-across-layers, {fl['n_seeds']} seeds): "
      f"mean={fl['mean']} p95={fl['p95']}")
print(f"  text floor ({' + '.join(tf['features'])}): AUC={tf['auc']} CI95={tf['auc_ci95']}")
print(f"  verdict: {r['verdict']}")
print(f"GATE 1 FINAL: probe best {r['best_layer']} AUC={best['auc']} "
      f"vs shuffled floor p95={fl['p95']} vs text floor AUC={tf['auc']}")
