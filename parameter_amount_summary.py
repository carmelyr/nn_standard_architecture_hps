"""
Model Parameter Statistics Generator

This script reads pareto_all_configurations.csv and computes the range and median
number of parameters for each dataset-model combination.

Outputs:
- param_stats.csv (summary)
- param_stats.md (Markdown tables)
"""

import pandas as pd

INPUT_FILE = "successive_halving_plots/pareto_analysis/pareto_all_configurations.csv"
OUTPUT_CSV = "param_stats.csv"
OUTPUT_MD = "param_stats.md"

# this function formats the range and median values for better readability
def format_range(row):
    min_val = row["min"]
    max_val = row["max"]
    median_val = row["median"]
    return f"{min_val:,} – {max_val:,} (median: {median_val:,})"

def main():
    # loads data
    df = pd.read_csv(INPUT_FILE)

    # groups by dataset and model, calculates min, max, median
    stats = (
        df.groupby(["Dataset", "Model"])["Model_Size_Parameters"]
          .agg(["min", "max", "median"])
          .astype(int)
          .reset_index()
    )

    # adds a formatted column for easier reading
    stats["Range_and_Median"] = stats.apply(format_range, axis=1)

    # saves results to CSV
    stats.to_csv(OUTPUT_CSV, index=False)

    # creates Markdown tables, one per dataset
    with open(OUTPUT_MD, "w") as f:
        f.write("# Parameter Statistics per Dataset and Model\n\n")
        for dataset, group in stats.groupby("Dataset"):
            f.write(f"## {dataset}\n\n")
            f.write("| Model | Min | Max | Median | Range & Median |\n")
            f.write("|-------|-----|-----|--------|----------------|\n")
            for _, row in group.iterrows():
                f.write(f"| {row['Model']} | {row['min']:,} | {row['max']:,} | {row['median']:,} | {row['Range_and_Median']} |\n")
            f.write("\n")

    print(f"Saved results to {OUTPUT_CSV} and {OUTPUT_MD}")

if __name__ == "__main__":
    main()
