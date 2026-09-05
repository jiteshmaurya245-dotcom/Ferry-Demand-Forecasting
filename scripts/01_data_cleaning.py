"""
01_data_cleaning.py

Loads the raw Toronto Island Ferry ticket dataset and produces a complete,
gap-free, outlier-capped dataset ready for feature engineering.

Key finding used here: not a single raw row has BOTH Sales Count and
Redemption Count equal to zero. This means the source system only logs an
interval when at least one transaction occurs -- so every "missing" 15-minute
slot represents a real interval of zero activity, not lost data. We exploit
that to safely fill every gap with (0, 0) rather than interpolating.

Usage:
    python 01_data_cleaning.py --input data/Toronto_Island_Ferry_Tickets.csv --output data/ferry_tickets_cleaned.csv
"""
import argparse
import pandas as pd


def clean(input_path: str, output_path: str) -> None:
    df = pd.read_csv(input_path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)

    # --- Sanity check the "implicit zero" hypothesis before relying on it ---
    both_zero = ((df["Sales Count"] == 0) & (df["Redemption Count"] == 0)).sum()
    print(f"Rows with both Sales and Redemption = 0: {both_zero} / {len(df)}")
    if both_zero > 0:
        print("WARNING: explicit zero-zero rows exist -- the implicit-zero "
              "assumption below may not hold for this dataset revision.")

    # --- Rebuild the full 15-minute grid and fill true gaps with zero ---
    full_grid = pd.date_range(df["Timestamp"].min(), df["Timestamp"].max(), freq="15min")
    full_df = pd.DataFrame({"Timestamp": full_grid})
    merged = full_df.merge(df[["Timestamp", "Sales Count", "Redemption Count"]], on="Timestamp", how="left")
    merged["Sales Count"] = merged["Sales Count"].fillna(0).astype(int)
    merged["Redemption Count"] = merged["Redemption Count"].fillna(0).astype(int)
    print(f"Rows before: {len(df)}  ->  after filling full grid: {len(merged)}")

    # --- Cap extreme outliers at the 99.9th percentile, keep raw values ---
    sales_cap = df["Sales Count"].quantile(0.999)
    redeem_cap = df["Redemption Count"].quantile(0.999)
    merged["Sales Count Raw"] = merged["Sales Count"]
    merged["Redemption Count Raw"] = merged["Redemption Count"]
    merged["is_outlier_capped"] = (merged["Sales Count"] > sales_cap) | (merged["Redemption Count"] > redeem_cap)
    merged["Sales Count"] = merged["Sales Count"].clip(upper=sales_cap).astype(int)
    merged["Redemption Count"] = merged["Redemption Count"].clip(upper=redeem_cap).astype(int)

    n_capped = merged["is_outlier_capped"].sum()
    print(f"Sales cap (99.9th pct): {sales_cap:.1f}  |  Redemption cap: {redeem_cap:.1f}")
    print(f"Rows capped: {n_capped} ({n_capped/len(merged)*100:.3f}%)")

    merged.to_csv(output_path, index=False)
    print(f"Saved cleaned dataset -> {output_path}  (shape: {merged.shape})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/Toronto_Island_Ferry_Tickets.csv")
    parser.add_argument("--output", default="data/ferry_tickets_cleaned.csv")
    args = parser.parse_args()
    clean(args.input, args.output)
