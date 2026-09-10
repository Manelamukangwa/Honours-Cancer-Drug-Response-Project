"""
Normalisation plots of GDSC2 and CTRPv2
"""


import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd
import os

# Folder containing the input CSVs
Input_dir = (
    "/Users/manellamukangwa/Honours-Cancer-Drug-Response-Project/"
    "Projects/Cancer-Drug-Response-Prediction/results/normalised"
)

# Output folder (same as input, per your setup)
Output_dir = Input_dir
os.makedirs(Output_dir, exist_ok=True)

# Each dataset has its own file + response column
datasets = {
    "GDSC2": {
        "path": os.path.join(Input_dir, "GDSC2.csv"),
        "response_col": "LN_IC50_curvecurator",
        "color": "tab:blue",
    },
    "CTRPv2_normalised": {
        "path": os.path.join(Input_dir, "CTRPv2_normalised.csv"),
        "response_col": "LN_IC50_curvecurator_normalised",
        "color": "tab:orange",
    },
}

# Combined Q-Q plot: one figure, both datasets overlaid
fig, ax = plt.subplots(figsize=(7, 7))

# Combined text file: all results written to one file
results_path = os.path.join(Output_dir, "normality_test_results.txt")
with open(results_path, "w") as f:

    for name, info in datasets.items():
        df = pd.read_csv(info["path"])
        response = df[info["response_col"]].dropna()

        # --- Q-Q plot data (compute manually so we can overlay on shared axes) ---
        (osm, osr), (slope, intercept, r) = stats.probplot(response, dist="norm")
        ax.scatter(osm, osr, s=10, alpha=0.5, color=info["color"], label=f"{name}")
        ax.plot(osm, slope * osm + intercept, color=info["color"], linestyle="--", linewidth=1)

        # --- D'Agostino-Pearson test ---
        stat, p = stats.normaltest(response)

        # --- Skewness and excess kurtosis ---
        skewness = stats.skew(response)
        kurtosis = stats.kurtosis(response)  # excess kurtosis (0 = normal)

        # --- Outlier rate (IQR rule) ---
        q1 = response.quantile(0.25)
        q3 = response.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outlier_count = ((response < lower_bound) | (response > upper_bound)).sum()
        n_observations = len(response)
        outlier_rate = outlier_count / n_observations

        f.write(f"D'Agostino-Pearson test — {name} response\n")
        f.write(f"column          = {info['response_col']}\n")
        f.write(f"n               = {len(response)}\n")
        f.write(f"statistic       = {stat:.4f}\n")
        f.write(f"p-value         = {p:.4f}\n")
        f.write(f"Normal (p >= 0.05)? {'Yes' if p >= 0.05 else 'No'}\n")
        f.write(f"skewness        = {skewness:.4f}\n")
        f.write(f"excess kurtosis = {kurtosis:.4f}\n")
        f.write(f"outlier count   = {outlier_count}\n")
        f.write(f"outlier rate    = {outlier_rate:.4f}\n")
        f.write("\n" + "-" * 40 + "\n\n")

        print(f"{name}: stat={stat:.4f}, p={p:.4f}, skew={skewness:.4f}, kurtosis={kurtosis:.4f}, outlier_rate={outlier_rate:.4f}")

ax.set_title("Q-Q Plot: GDSC2 vs CTRPv2_normalised")
ax.set_xlabel("Theoretical quantiles")
ax.set_ylabel("Ordered values")
ax.legend()
fig.savefig(os.path.join(Output_dir, "GDSC2_vs_CTRPv2_qq_plot.png"), dpi=300, bbox_inches="tight")
plt.close(fig)

print(f"Saved combined plot and results to {Output_dir}")