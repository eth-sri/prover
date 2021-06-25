import R2
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor
from tqdm import tqdm
import time
import os, sys, random
from models_mnist import MnistModel, MnistModelDP
from utils import get_default_device
import argparse

parser = argparse.ArgumentParser(description="MNIST verification")
parser.add_argument(
    "--nframes",
    type=int,
    default=4,
    choices=[4, 7, 14, 28],
    help="the number of fixed frames. choose between 4, 7, 14, and 28.",
)
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

nframes = args.nframes
nhidden = args.nhidden
nlayers = args.nlayers
eps = args.eps
bound_method = args.bound_method
seed = args.seed
model_name = args.model_dir

inp_size = int(28 * 28 / nframes)

dev = get_default_device()
devtxt = "cuda" if dev == torch.device("cuda") else "cpu"

stt_dict = torch.load(model_name, map_location=devtxt)

model = MnistModel(inp_size, nhidden, nlayers, 10).to(dev)
model.load_state_dict(torch.load(model_name, map_location=devtxt))

r2model = MnistModelDP(inp_size, nhidden, nlayers, 10).to(dev)
r2model.load_state_dict(torch.load(model_name, map_location=devtxt))

r2model.set_bound_method(bound_method)

dataset = MNIST(root="data/", train=False, download=True, transform=ToTensor())
gen = torch.Generator()
gen.manual_seed(seed)
indices = torch.randperm(len(dataset), generator=gen).tolist()
dataset = [(x[0][0].view(784), x[1]) for x in dataset]

proven_dp = 0
correct = 0
running_time = 0.0
for i in tqdm(range(120)):
    print(f"[Testing Input #{i:03d} ({proven_dp} proven / {correct} correct)]")
    x, y = dataset[indices[i]]
    input = x.unsqueeze(0).to(dev)
    res = model(input)

    _, pred_label = torch.max(res[0], 0)
    if pred_label.item() != y:
        print(" - prediction failed")
        continue

    correct += 1

    input_dp = R2.DeepPoly.deeppoly_from_perturbation(x.to(dev), eps, truncate=(0, 1))
    st = time.time()
    proven = r2model.certify(input_dp, y, verbose=args.verbose)
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
res_name = f"results/exp_mnist_{nframes}f_{nhidden}h_{nlayers}l.csv"
with open(res_name, "a") as f:
    f.write(f"{bound_method},{eps},{proven_dp/correct*100},{running_time/correct}\n")
