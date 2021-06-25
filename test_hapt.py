import torch
import time
import os, sys
import numpy as np
from models_mnist import MnistModel, MnistModelDP
from tqdm import tqdm

import R2
import hapt_loader
from utils import get_default_device
import argparse


parser = argparse.ArgumentParser(description="HAPT verification")
parser.add_argument(
    "--nhidden", type=int, default=32, help="the hidden dimension of each LSTM cell."
)
parser.add_argument("--nlayers", type=int, default=1, help="the number of LSTM layers.")
parser.add_argument("--eps", type=float, default=0.01, help="perturbation epsilon.")
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

hidden_dim = args.nhidden
num_layers = args.nlayers
eps = args.eps
bound_method = args.bound_method
seed = args.seed
model_name = args.model_dir
# model_name = f"saved/hapt_{num_layers}L_{hidden_dim}H.pt"

dev = get_default_device()
devtxt = "cuda" if dev == torch.device("cuda") else "cpu"
print(devtxt)

model = MnistModel(75, hidden_dim, num_layers, 6).to(dev)
model.load_state_dict(torch.load(model_name, map_location=devtxt))
r2model = MnistModelDP(75, hidden_dim, num_layers, 6).to(dev)
r2model.load_state_dict(torch.load(model_name, map_location=devtxt))
r2model.set_bound_method(bound_method)

print("Test data loading")
loader = hapt_loader.DataStream("test", batch=1, seed=seed)

proven_dp = 0
correct = 0
running_time = 0.0
i = 0
for x, y in loader:
    print(f"[Testing Input #{i:03d} ({proven_dp} proven / {correct} correct)]")
    i += 1
    input = torch.Tensor(x).to(dev)
    out = model(input)

    _, pred_label = torch.max(out[0], 0)
    if pred_label.item() != y[0]:
        print(" - prediction failed")
        continue

    correct += 1

    input_dp = R2.DeepPoly.deeppoly_from_perturbation(input[0], eps)
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
res_name = f"results/exp_hapt.csv"
with open(res_name, "a") as f:
    f.write(f"{bound_method},{eps},{proven_dp/correct*100},{running_time/correct}\n")
