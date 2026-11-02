# ---- 28. Revision: equal-budget controls for the shortcut repair (MNIST-EvenOdd, Theorem 8). Same sum-only training as cell 19b,
#      on five training sets of the same size: the original support (base), and the original support plus 10% extra sum-labelled
#      pairs of one of four kinds: the Theorem 8 repair 0+1 (leaves 1 map), the non-repairing cross pair 0+5 (leaves 6 maps), the
#      non-repairing within-class pair 0+2 (leaves 6 maps), or more pairs from the original support (leaves 36 maps). For every
#      head: digit error on all pairs, in-distribution task accuracy, the shortcut criterion of the paper, and how well ONE
#      deterministic digit map explains its readings (map purity), so the deterministic-map model of Theorem 8 is checked.
#      Resumable per encoder. Writes s28_evenodd_controls.json to the work dir. GPU, a few minutes ----
import os as os_28
import numpy as np_28

CFG_28 = dict(extra=int(round(V14["n_pairs_train"] * V14["repair_share"])), seeds=tuple(V14["nesy_seeds"]), file="s28_evenodd_controls.json")
EXTRA_28 = {"repair_0+1": [(0, 1), (1, 0)], "nonrepair_0+5": [(0, 5), (5, 0)], "nonrepair_0+2": [(0, 2), (2, 0)], "more_support": list(V14_ALLOWED)}
TRAIN_28 = {"base": V14_IDX["train"]}
for name_28, pairs_28 in EXTRA_28.items():
    TRAIN_28[name_28] = np_28.concatenate([V14_IDX["train"], v14_make_pairs(V14_YTR, CFG_28["extra"], pairs_28, V14["seed"] + 5)])
MAPS_28 = {k_: int(v14_count_rs(sorted(set(V14_ALLOWED) | set(p_ for p_ in EXTRA_28.get(k_, []))))) for k_ in TRAIN_28}
print("training sets:", {k_: len(v_) for k_, v_ in TRAIN_28.items()}, "| deterministic maps left (Theorem 8):", MAPS_28)


def map_stats_28(pred, true):
    """Best single deterministic map alpha(d) = most frequent reading of digit d, the share of test digits it explains (purity),
    whether it is the identity, and whether it is consistent with the original training support."""
    alpha = [int(np_28.bincount(pred[true == d], minlength=10).argmax()) if (true == d).any() else d for d in range(10)]
    purity = float(np_28.mean(pred == np_28.asarray(alpha)[true]))
    cons = all(alpha[a] + alpha[b] == a + b for a, b in V14_ALLOWED)
    return {"alpha": alpha, "purity": purity, "identity": alpha == list(range(10)), "consistent_with_support": bool(cons)}


RES_28 = load_json(ckpt(CFG_28["file"])) if os_28.path.exists(ckpt(CFG_28["file"])) else {"maps_left": MAPS_28, "extra_pairs": CFG_28["extra"], "heads": {}}


def run_encoder_28(short, Ftr, Fte, res):
    """Train every condition x seed for one encoder (skipping heads already in res) and add their scores to res['heads']."""
    Xtr = torch.as_tensor(Ftr, device=DEV); Xte = torch.as_tensor(Fte, device=DEV); mu = Xtr.mean(0)
    Xc = (Xtr - mu) * 10; Xtc = (Xte - mu) * 10; tid, tall = V14_IDX["test_id"], V14_IDX["test_all"]
    for cond, idx in TRAIN_28.items():
        s = torch.as_tensor(V14_YTR[idx].sum(1), device=DEV)
        for sd in CFG_28["seeds"]:
            key = f"{short}:{cond}:s{sd}"
            if key in res["heads"]:
                continue
            W, b = v14_nesy(Xc[idx[:, 0]], Xc[idx[:, 1]], s, None, V14["seed"] + sd)
            with torch.no_grad():
                dte = (Xtc @ W.T + b).argmax(1).cpu().numpy()
            pa, ya, pi, yi = dte[tall], V14_YTE[tall], dte[tid], V14_YTE[tid]
            err = float(np_28.mean(pa != ya)); acc = float(np_28.mean(pi.sum(1) == yi.sum(1)))
            res["heads"][key] = {"encoder": short, "condition": cond, "seed": int(sd), "digit_err_all": err, "task_acc_id": acc,
                                 "shortcut": bool(acc >= 0.7 and err >= 0.2), **map_stats_28(dte, V14_YTE)}


for _enc, (_Ftr, _Fte, _T) in v14_feats().items():
    _short = _enc.rstrip("/").split("/")[-1]
    if all(f"{_short}:{_c}:s{_s}" in RES_28["heads"] for _c in TRAIN_28 for _s in CFG_28["seeds"]):
        print(f"{_short}: done already"); continue
    run_encoder_28(_short, _Ftr, _Fte, RES_28); save_json(ckpt(CFG_28["file"]), RES_28); gpu_free()
    print(f"{_short}: " + ", ".join(f"{_c} {np_28.mean([RES_28['heads'][f'{_short}:{_c}:s{_s}']['digit_err_all'] for _s in CFG_28['seeds']]):.3f}" for _c in TRAIN_28), flush=True)

SUMMARY_28 = {}
for _cond in TRAIN_28:
    hs_28 = [h_ for h_ in RES_28["heads"].values() if h_["condition"] == _cond]
    if not hs_28:
        continue
    e_28 = np_28.array([h_["digit_err_all"] for h_ in hs_28])
    SUMMARY_28[_cond] = {"heads": len(hs_28), "maps_left": MAPS_28[_cond], "digit_err_mean": float(e_28.mean()),
                           "digit_err_ci95": float(1.96 * e_28.std(ddof=1) / np_28.sqrt(len(e_28))) if len(e_28) > 1 else 0.0,
                           "digit_err_by_seed": {str(s_): float(np_28.mean([h_["digit_err_all"] for h_ in hs_28 if h_["seed"] == s_])) for s_ in CFG_28["seeds"]},
                           "task_acc_id_mean": float(np_28.mean([h_["task_acc_id"] for h_ in hs_28])),
                           "shortcut_share": float(np_28.mean([h_["shortcut"] for h_ in hs_28])),
                           "map_purity_mean": float(np_28.mean([h_["purity"] for h_ in hs_28])),
                           "map_purity_of_shortcut_heads": float(np_28.mean([h_["purity"] for h_ in hs_28 if h_["shortcut"]])) if any(h_["shortcut"] for h_ in hs_28) else None,
                           "shortcut_heads_with_support_consistent_map": int(sum(h_["shortcut"] and h_["consistent_with_support"] and not h_["identity"] for h_ in hs_28)),
                           "shortcut_heads": int(sum(h_["shortcut"] for h_ in hs_28))}
RES_28["summary"] = SUMMARY_28
save_json(ckpt(CFG_28["file"]), RES_28)
print("\ncondition       maps  heads  digit err (mean +- 95%)  task acc ID  shortcut share  map purity  shortcut heads with a support-consistent non-identity map")
for _c, _v in SUMMARY_28.items():
    print(f"{_c:15s} {_v['maps_left']:4d}  {_v['heads']:5d}  {_v['digit_err_mean']:.3f} +- {_v['digit_err_ci95']:.3f}        {_v['task_acc_id_mean']:.3f}        "
          f"{_v['shortcut_share']:.2f}            {_v['map_purity_mean']:.3f}       {_v['shortcut_heads_with_support_consistent_map']} of {_v['shortcut_heads']}")
hf_push(RUN_NAME, force=True)
