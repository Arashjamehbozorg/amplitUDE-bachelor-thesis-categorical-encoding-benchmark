# combine_ames_results.py
import pandas as pd
from pathlib import Path

from encoding_benchmark_defs import EncodingBenchmark, RESULTS_DIR

def main():
    dataset_name = "Ames_Housing"

    # 1) Load all scenario CSVs
    csv_paths = sorted(RESULTS_DIR.glob(f"{dataset_name}_scenario_*_benchmark_results.csv"))
    if not csv_paths:
        raise SystemExit("No scenario CSVs found. Make sure all 4 scenario jobs finished.")

    dfs = [pd.read_csv(p) for p in csv_paths]
    combined = pd.concat(dfs, ignore_index=True)

    # 2) Save combined CSV (optional but nice)
    combined_csv = RESULTS_DIR / f"{dataset_name}_combined_benchmark_results.csv"
    combined.to_csv(combined_csv, index=False)
    print(f"✓ Combined results saved to: {combined_csv}")

    # 3) Run analysis + visualization on combined data
    bench = EncodingBenchmark(random_state=42)

    print("\n" + "📊 COMBINED ANALYSIS".center(70, "-"))
    bench.analyze_results(combined)

    print("\n" + "📈 COMBINED VISUALIZATION".center(70, "-"))
    # This call creates the colourful PNG with all scenarios
    bench.visualize_results(combined, dataset_name=dataset_name)

if __name__ == "__main__":
    main()
