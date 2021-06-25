import torch
import time
import os, sys
import numpy as np
from models_speech import SpeechClassifier, SpeechClassifierDP
from tqdm import tqdm

import R2
import gsc_loader
import fsdd_loader
from utils import get_default_device
import argparse

parser = argparse.ArgumentParser(description="FSDD and GSC verification")
parser.add_argument(
    "--dataset", type=str, choices=["FSDD", "GSC"], help="dataset either FSDD or GSC."
)
parser.add_argument("--db", type=float, default=-90, help="perturbation decibel.")
parser.add_argument(
    "--bound_method",
    type=str,
    default="lp",
    choices=["lp", "opt"],
    help="bounding method, either lp or opt.",
)
parser.add_argument("--model_dir", type=str, default="", help="target model directory.")
parser.add_argument(
    "--seed", type=int, default=1000, help="random seed for reproducibility."
)
parser.add_argument(
    "--verbose", action="store_true", help="print debug information during verifiation."
)
args = parser.parse_args()

dataset = args.dataset
eps = args.db
bound_method = args.bound_method
seed = args.seed
model_name = args.model_dir

dev = get_default_device()
devtxt = "cuda" if dev == torch.device("cuda") else "cpu"
print(devtxt)

if dataset == "GSC":
    model = SpeechClassifier(1024, 1024, 10, 50, 50, 2, 35, 1, 8000).to(dev)
    model.load_state_dict(torch.load(model_name, map_location=devtxt))
    r2model = SpeechClassifierDP(1024, 1024, 10, 50, 50, 2, 35, 1, 8000).to(dev)
    r2model.load_state_dict(torch.load(model_name, map_location=devtxt))
    r2model.set_bound_method(bound_method)

    print("Test data loading")
    loader = gsc_loader.DataStream("test", batch=1, seed=seed)

elif dataset == "FSDD":
    model = SpeechClassifier(256, 200, 10, 40, 32, 2, 10, 1, 8000).to(dev)
    model.load_state_dict(torch.load(model_name, map_location=devtxt))
    r2model = SpeechClassifierDP(256, 200, 10, 40, 32, 2, 10, 1, 8000).to(dev)
    r2model.load_state_dict(torch.load(model_name, map_location=devtxt))
    r2model.set_bound_method(bound_method)

    print("Test data loading")
    loader = fsdd_loader.DataStream("test", batch=1, seed=seed)

proven_dp = 0
correct = 0
running_time = 0.0
i = 0
for x, y in loader:
    print(f"[Testing Input #{i:03d} ({proven_dp} proven / {correct} correct)]")
    i += 1
    # input = torch.Tensor(x).unsqueeze(0).to(dev)
    input = torch.Tensor(x).to(dev)
    out = model(input)

    _, pred_label = torch.max(out[0], 0)
    if pred_label.item() != y[0]:
        print(" - prediction failed")
        continue

    correct += 1

    input_dp = R2.DeepPoly.deeppoly_from_dB_perturbation(input[0], eps)
    st = time.time()
    proven = r2model.certify(input_dp, y[0], verbose=args.verbose)
    ed = time.time()
    running_time += ed - st

    print(f" - took {ed-st} sec to verify")
    if proven:
        print("\t[PROVEN]")
        proven_dp += 1
    else:
        print("\t[FAILED]")
    if correct == 100:
        break

print(f"provability: {proven_dp/correct*100}%")
print(f"avg running time: {running_time/correct}")

os.makedirs("results", exist_ok=True)
res_name = "results/exp_speech.csv"
with open(res_name, "a") as f:
    f.write(
        f"{dataset},{bound_method},{eps},{proven_dp/correct*100},{running_time/correct}\n"
    )
