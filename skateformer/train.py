import argparse
import shutil
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--data_dt", type=str)
parser.add_argument("--dt", type=str)
args = parser.parse_args()

# data_dt = "20251128_1848"
# dt      = "20251128_1911"
data_dt = args.data_dt
dt = args.dt
datapath = Path(f"/media/Eason/Aldo/SkateFormer/data/sarcopenia/{data_dt}")
config = "./config/sarcopenia/train_base.yaml"
base_work = Path(f"./work_dir/{dt}")
base_work.mkdir()

# Copy base config
shutil.copyfile(config, base_work / "base_config.yaml")

for npz in sorted(datapath.glob("*.npz")):
    part = npz.stem  # e.g., "partition0"
    
    # Copy partition data
    shutil.copyfile(datapath / (part + ".csv"), base_work / (part + ".csv"))
    # Copy partition description
    shutil.copyfile(datapath / (part + ".txt"), base_work / (part + ".txt"))

    train_work = base_work / part
    train_work.mkdir(exist_ok=True)
    subprocess.run([
        "python", "main.py",
        "--config", config,
        "--work-dir", str(train_work),
        "--data_path", str(npz)
    ], check=True)

    # test with last epoch model
    weight_glob = list(train_work.glob("runs-55-*.pt"))[0]
    test_work = base_work / (part + "_test")
    test_work.mkdir(exist_ok=True)
    subprocess.run([
        "python", "main.py",
        "--config", config,
        "--phase", "test",
        "--ignore-weights", "[]",
        "--weights", weight_glob,
        "--work-dir", str(test_work),
        "--data_path", str(npz)
    ], check=True)
