# run_ames.py — entrypoint for amplitUDE benchmark job

import os, time, matplotlib
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    module="xgboost.core"
)


matplotlib.use("Agg")  # disable GUI backend on HPC

from encoding_benchmark_defs import load_ames_data, EncodingBenchmark

def main(scenario_id: int):
    WS = Path("/lustre/scratch/soarjame-bachelor-thesis")
    CODE = WS / "code" / "Boruta_Main_Amplitude"
    RESULTS = CODE / "results"
    RESULTS.mkdir(parents=True, exist_ok=True)

    # point load_ames_data() to the dataset path
    os.environ.setdefault("DATA_ROOT", str(CODE / "data"))

    # ---- load dataset ----
    loaded = load_ames_data()
    if loaded is None:
        raise SystemExit("❌ Dataset not found. Make sure AmesHousing.csv is in data/raw/ames-housing-dataset/")
    X, y, cat_cols, num_cols = loaded
    print(f"✓ Loaded Ames dataset: X={X.shape}, y={y.shape}, #cat={len(cat_cols)}, #num={len(num_cols)}")

    # ---- run benchmark ----
    bench = EncodingBenchmark(random_state=42)
    t0 = time.time()
    results = bench.run_scenarios(X, y, cat_cols, num_cols, scenario_id = scenario_id)
    print(f"✓ Benchmark finished in {(time.time()-t0)/60:.1f} minutes.")

    # ---- save results ----
    if scenario_id is None:
        out_csv = RESULTS / "ames_four_scenarios_results.csv"
    else:
        out_csv = RESULTS / f"ames_scenario_{scenario_id}_results.csv"
    
    results.to_csv(out_csv, index=False)
    print(f"✓ Results saved to {out_csv}")

    # ---- analysis + plots ----
    bench.analyze_results(results)
    bench.visualize_results(results, dataset_name=f"ames_scen{scenario_id}" if scenario_id is not None else "ames")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario-id",
        type=int,
        default=None,
        help="0–3. If omitted, run all 4 scenarios."
    )
    args = parser.parse_args()

    main(scenario_id=args.scenario_id)
