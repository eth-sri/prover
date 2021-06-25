import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from utils import get_default_device
import R2
import time


class MnistModel(nn.Module):
    def __init__(self, in_size, hidden_size, num_layers, out_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.out_size = out_size

        self.frame_size = in_size
        self.frame_step = in_size

        dev = get_default_device()
        self.lstm = nn.LSTM(in_size, hidden_size, num_layers).to(dev)
        self.linear = nn.Linear(hidden_size, out_size).to(dev)

    def forward(self, xb):
        frames = xb.unfold(1, self.frame_size, self.frame_step)
        frames = frames.transpose(0, 1)
        _, (out, _) = self.lstm(frames)
        out = out[-1]
        out = self.linear(out)
        return out


class MnistModelDP(MnistModel):
    def __init__(self, *args, **kwargs):
        super(MnistModelDP, self).__init__(*args, **kwargs)

    def set_bound_method(self, bound_method):
        self.bound_method = bound_method

    def certify(self, input, gt, max_iter=100, verbose=False):
        layers = []
        lstm_pack = []
        feed = []
        dev = input.device

        for frame_idx, i in enumerate(range(0, input.lb.size()[0], self.frame_step)):
            if i + self.frame_size > input.lb.size()[0]:
                break
            chain = []
            dpframe = R2.DeepPoly(
                input.lb[i : i + self.frame_size],
                input.ub[i : i + self.frame_size],
                None,
                None,
            )
            feed.append(dpframe)

            lin1 = R2.Linear(self.frame_size, self.frame_size)
            lin1.assign(torch.eye(self.frame_size), device=dev)
            lin1_out = lin1(dpframe)
            chain.append(lin1)

            lstm = R2.LSTMCell.convert(
                self.lstm,
                prev_layer=lin1,
                prev_cell=None if frame_idx == 0 else lstm_pack[-1],
                method=self.bound_method,
            )
            lstm_pack.append(lstm)
            lstm_out = lstm(lin1_out)
            chain.append(lstm)

            layers.append(chain)

        lin2 = R2.Linear.convert(self.linear, prev_layer=lstm_pack[-1], device=dev)
        lin2_out = lin2(lstm_out)
        layers.append(lin2)

        out_dim = lin2.out_features
        lin_compare = R2.Linear(out_dim, 1, prev_layer=layers[-1])
        layers.append(lin_compare)

        lp_proven = True
        for fl in range(out_dim):  # false label
            if fl == gt:
                continue
            if verbose:
                print(f"Testing label {fl} | ground truth {gt}")

            comp_mtx = torch.zeros(out_dim, 1)
            comp_mtx[gt, 0] = 1
            comp_mtx[fl, 0] = -1
            lin_compare.assign(comp_mtx, device=dev)
            lp_res = lin_compare(lin2_out)

            if lp_res.lb[0] > 0:
                if verbose:
                    print("\tProven.")
                continue
            elif self.bound_method != "opt":
                if verbose:
                    print("\tFailed.")
                return False

            lp_proven = False
            st_time = time.time()
            lmbs = []
            for chain in layers[:-2]:
                for layer in chain:
                    if hasattr(layer, "lmb"):
                        lmbs.append(layer.set_lambda(dev))

            optim = torch.optim.Adam(lmbs)
            lr_fn = lambda e: 100 * 0.98 ** e
            scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_fn)
            success = False
            for epoch in range(max_iter):
                for chain, inp in zip(layers[:-2], feed):
                    out = inp
                    for layer in chain:
                        out = layer(out)
                out = layers[-2](out)
                out = layers[-1](out)

                if verbose:
                    print(f"\tEpoch {epoch}: min(LB_gt - UB_fl) = {out.lb[0]}")
                if out.lb[0] > 0:
                    if verbose:
                        print("\tROBUSTNESS PROVED")
                    success = True
                    break

                loss = -out.lb[0]
                loss.backward(retain_graph=True)
                optim.step()
                optim.zero_grad()
                scheduler.step()

            if not success:
                return False

        return True
