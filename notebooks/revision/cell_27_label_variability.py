# ---- 27. Revision: spread of the label-budget results. Reads the saved per-draw records of the knowledge-stratified (KSE)
#      vs random labelling experiments (CUB s14c_v9, MNIST-Add s15c_v10, BDD-OIA s16d_v11_bdd, GTSRB s17c_v12_gtsrb), and reports
#      for every budget: mean regret, 95% interval of the mean over the 20 draws, and the share of draws in which KSE beat / tied /
#      lost to random labelling (paired, same draw). Redraws fig_labels with error bars. No training, CPU only, seconds.
#      Writes s27_label_variability.json and paper_figures/fig_labels.pdf/.png to the work dir ----
import os as os_27
import numpy as np_27
import matplotlib.pyplot as plt_27
from scipy.stats import t as t_27

SRC_27 = [("CUB birds", "s14c_v9.json", "curves", "KSE"), ("GTSRB signs", "s17c_v12_gtsrb.json", "kse", "KSE"),
          ("MNIST-Add", "s15c_v10.json", "kse", "KSE_prop"), ("BDD-OIA", "s16d_v11_bdd.json", "kse", "KSE")]


def draws_27(recs, key):
    """{m: (random array, KSE array)} with one value per draw. CUB records are per (concept, m, rep): average over concepts
    within each rep first, so a draw is one rep. Other files hold one record per (m, draw) in draw order."""
    out = {}
    for m in sorted({int(r["m"]) for r in recs}):
        rs = [r for r in recs if int(r["m"]) == m]
        if rs and "rep" in rs[0]:
            reps = sorted({int(r["rep"]) for r in rs})
            ra = [np_27.mean([r["random"] for r in rs if int(r["rep"]) == q]) for q in reps]
            ka = [np_27.mean([r[key] for r in rs if int(r["rep"]) == q]) for q in reps]
        else:
            ra = [r["random"] for r in rs]; ka = [r[key] for r in rs]
        out[m] = (np_27.asarray(ra, float), np_27.asarray(ka, float))
    return out


def ci_27(x):
    """Mean and half-width of the 95% t-interval of the mean."""
    n = len(x)
    if n < 2:
        return float(np_27.mean(x)), 0.0
    return float(np_27.mean(x)), float(t_27.ppf(0.975, n - 1) * np_27.std(x, ddof=1) / np_27.sqrt(n))


RES_27 = {}
for title_27, file_27, field_27, key_27 in SRC_27:
    if not os_27.path.exists(ckpt(file_27)):
        print(f"{title_27}: {file_27} not found, skipped"); continue
    recs_27 = load_json(ckpt(file_27)).get(field_27, [])
    if not recs_27:
        print(f"{title_27}: no per-draw records in {file_27}, skipped"); continue
    rows_27 = {}
    for m_27, (ra_27, ka_27) in draws_27(recs_27, key_27).items():
        mr_27, hr_27 = ci_27(ra_27); mk_27, hk_27 = ci_27(ka_27); d_27 = ka_27 - ra_27; md_27, hd_27 = ci_27(d_27)
        rows_27[str(m_27)] = {"draws": int(len(ra_27)), "random_mean": mr_27, "random_ci95": hr_27, "KSE_mean": mk_27, "KSE_ci95": hk_27,
                              "diff_mean": md_27, "diff_ci95": hd_27, "KSE_better": float(np_27.mean(d_27 < -1e-12)),
                              "tie": float(np_27.mean(np_27.abs(d_27) <= 1e-12)), "KSE_worse": float(np_27.mean(d_27 > 1e-12))}
    RES_27[title_27] = rows_27

print("label budget | random: mean +- 95% CI | KSE: mean +- 95% CI | KSE-random: mean +- CI | KSE better / tie / worse (share of draws)")
for title_27, rows_27 in RES_27.items():
    print(f"== {title_27}")
    for m_27, r_27 in rows_27.items():
        print(f"  m={m_27:>4s} ({r_27['draws']} draws) | {r_27['random_mean']:.5f} +- {r_27['random_ci95']:.5f} | {r_27['KSE_mean']:.5f} +- {r_27['KSE_ci95']:.5f}"
              f" | {r_27['diff_mean']:+.5f} +- {r_27['diff_ci95']:.5f} | {r_27['KSE_better']:.2f} / {r_27['tie']:.2f} / {r_27['KSE_worse']:.2f}")
save_json(ckpt("s27_label_variability.json"), RES_27)

FIG_DIR_27 = ckpt("paper_figures"); os_27.makedirs(FIG_DIR_27, exist_ok=True)
plt_27.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})
PANELS_27 = [t_ for t_ in ("CUB birds", "GTSRB signs", "MNIST-Add") if t_ in RES_27]
if PANELS_27:
    fig_27, ax_27 = plt_27.subplots(1, len(PANELS_27), figsize=(2.4 * len(PANELS_27), 2.3), squeeze=False)
    for a_27, title_27 in zip(ax_27[0], PANELS_27):
        rows_27 = RES_27[title_27]; ms_27 = [int(m) for m in rows_27]
        for lab_27, k_27, col_27, mk_27 in (("random labels", "random", "#8c8c8c", "o"), ("knowledge-stratified", "KSE", "#1f6fb2", "s")):
            mu_27 = np_27.array([rows_27[str(m)][f"{k_27}_mean"] for m in ms_27]); h_27 = np_27.array([rows_27[str(m)][f"{k_27}_ci95"] for m in ms_27])
            a_27.errorbar(ms_27, mu_27, yerr=h_27, fmt=mk_27 + "-", color=col_27, ms=3, lw=1, capsize=2, label=lab_27)
        a_27.set_xscale("log"); a_27.set_title(title_27, fontsize=8); a_27.set_xlabel("labels")
    ax_27[0][0].set_ylabel("selection regret"); ax_27[0][0].legend(fontsize=6, frameon=False)
    fig_27.tight_layout()
    for ext_27 in ("pdf", "png"):
        fig_27.savefig(os_27.path.join(FIG_DIR_27, f"fig_labels.{ext_27}"), bbox_inches="tight")
    plt_27.close(fig_27)
    print(f"fig_labels redrawn with 95% intervals: {os_27.path.join(FIG_DIR_27, 'fig_labels.pdf')}")
hf_push(RUN_NAME, force=True)
