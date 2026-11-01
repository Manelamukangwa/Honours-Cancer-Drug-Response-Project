"""
Paired R² differences (RF, XGBoost, Neural Network) across LCO splits, one panel per dataset.
Reads bootstrap_results.csv (rows with metric == "R2").

Sign convention: first-named model minus second-named; > 0 means the first-named model is better.
  RF - XGBoost, RF - Neural Network : paired bootstrap CIs from the file ("model minus RF", sign flipped)
  XGBoost - Neural Network          : point estimate only (no paired CI in the file)
"""
import sys
import pandas as pd
import matplotlib.pyplot as plt

csv_path = sys.argv[1] if len(sys.argv) > 1 else "bootstrap_results.csv"
out_stem = sys.argv[2] if len(sys.argv) > 2 else "fig_paired_r2_differences"
SHARE_X = True

DATASETS = ["CCLE", "CTRPv1", "GDSC1", "GDSC2"]
COMPS = ["RF − XGBoost", "RF − Neural Network", "XGBoost − Neural Network"]
COL = dict(zip(COMPS, ["#E69F00", "#009E73", "#0072B2"]))
OFF = dict(zip(COMPS, [-0.24, 0.0, 0.24]))

d = pd.read_csv(csv_path)
d = d[d.metric == "R2"]
diff = d[d.comparison == "minus RF"].pivot(index=["dataset", "split"], columns="model",
                                           values=["estimate", "ci_low", "ci_high"])

rows = []
for (ds, sp), r in diff.iterrows():
    # model - RF  ->  RF - model: negate estimate, swap and negate CI bounds
    rows.append((ds, sp, COMPS[0], -r[("estimate", "XGB")], -r[("ci_high", "XGB")], -r[("ci_low", "XGB")]))
    rows.append((ds, sp, COMPS[1], -r[("estimate", "NN")], -r[("ci_high", "NN")], -r[("ci_low", "NN")]))
    # XGB - NN = (XGB - RF) - (NN - RF); point estimate only
    xn = r[("estimate", "XGB")] - r[("estimate", "NN")]
    rows.append((ds, sp, COMPS[2], xn, float("nan"), float("nan")))
res = pd.DataFrame(rows, columns=["dataset", "split", "comparison", "diff", "lo", "hi"])
res.to_csv(f"{out_stem}_data.csv", index=False)

fig, axes = plt.subplots(2, 2, figsize=(10, 8.5), sharex=SHARE_X)
for ax, ds in zip(axes.ravel(), DATASETS):
    for c in COMPS:
        s = res[(res.dataset == ds) & (res.comparison == c)].sort_values("split")
        y = s["split"].to_numpy() + OFF[c]
        if c != COMPS[2]:
            ax.errorbar(s["diff"], y, xerr=[s["diff"] - s["lo"], s["hi"] - s["diff"]],
                        fmt="o", ms=4.5, lw=1.4, capsize=2.5, color=COL[c])
        else:
            ax.plot(s["diff"], y, "D", ms=4.5, mfc="white", mec=COL[c], mew=1.4, ls="")
    ax.axvline(0, color="black", lw=1)
    ax.grid(axis="x", color="0.9", lw=0.8); ax.set_axisbelow(True)
    ax.set_yticks(range(1, 6)); ax.set_ylim(5.6, 0.4)
    ax.set_title(ds, fontweight="bold", fontsize=11); ax.set_ylabel("LCO split")
    for sp in ("top", "right"): ax.spines[sp].set_visible(False)
for ax in axes[1]: ax.set_xlabel("Difference in R²")

h = [plt.Line2D([], [], marker="o", color=COL[c], label=c + " (95% CI)") for c in COMPS[:2]]
h.append(plt.Line2D([], [], marker="D", mfc="white", mec=COL[COMPS[2]], color=COL[COMPS[2]], ls="",
                    label=COMPS[2] + " (point estimate)"))
fig.legend(handles=h, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0), fontsize=9)
fig.text(0.5, 0.945, "> 0: first-named model better", ha="center", fontsize=9, color="0.3")
fig.tight_layout(rect=(0, 0, 1, 0.93))
fig.savefig(f"{out_stem}.pdf", bbox_inches="tight")
fig.savefig(f"{out_stem}.png", dpi=200, bbox_inches="tight")
print(res.round(3).to_string())