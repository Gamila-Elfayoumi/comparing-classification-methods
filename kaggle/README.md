# Kaggle handoff

The notebook `run_benchmark.ipynb` is ready to upload to Kaggle. Add these two public input datasets to the notebook:

- `kmader/skin-cancer-mnist-ham10000`
- `alitaqishah/blood-cell-anomaly-detection-2025`

Also attach this repository as a Kaggle Dataset or clone its GitHub URL when notebook internet access is enabled. Set `PROJECT` in the first code cell to the attached source directory. Run all cells. The notebook copies only the three required CSVs into Kaggle's writable directory, installs dependencies, executes the benchmark, and displays summary charts. Its source includes no Fashion-MNIST.

Kaggle account publication needs the account owner's authentication and a final check of dataset licensing and notebook visibility. The local notebook is a publication-ready handoff, not a claim that a Kaggle page already exists.
