import sys, json, pickle
c = pickle.load(open(sys.argv[1], "rb"))
print("epoch", c["epoch"], "step", c["step"], "best", c["best_val_loss"])
print("metrics", {k: (round(float(v), 6) if not isinstance(v, dict) else {kk: round(float(vv), 6) for kk, vv in list(v.items())[:6]}) for k, v in c["metrics"].items()} if isinstance(c["metrics"], dict) else c["metrics"])
cfg = json.load(open(sys.argv[2])); cc = c["config"]
print("config diffs:", {k: (cc.get(k), cfg.get(k)) for k in set(cc) | set(cfg) if cc.get(k) != cfg.get(k)})
