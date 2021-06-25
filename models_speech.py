import torch, math, time
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from utils import M, get_default_device
import R2
import fsdd_loader


class SpeechClassifier(nn.Module):
    def __init__(
        self,
        frame_size=256,
        frame_step=200,
        n_filt=10,
        hidden_dim=40,
        hidden_dim2=32,
        num_layers=2,
        out_dim=10,
        batch_size=5,
        samprate=8000,
    ):
        super().__init__()
        dev = get_default_device()
        m = M(frame_size, frame_step, n_filt, samprate, dev)

        self.frame_size = frame_size
        self.frame_step = frame_step
        self.n_filt = n_filt
        self.hidden_dim = hidden_dim
        self.hidden_dim2 = hidden_dim2
        self.out_dim = out_dim
        self.batch_size = batch_size

        self.mtx1 = (m.preemph * m.hamming) @ m.W_comb
        self.mtx2 = m.real_add_img @ m.fb

        self.linear1 = nn.Linear(n_filt, hidden_dim).to(dev)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim2, num_layers).to(dev)
        self.linear2 = nn.Linear(hidden_dim2, out_dim).to(dev)

    def forward(self, input):
        if input.size()[1] % self.frame_step != 0:
            input = torch.cat(
                (input, torch.zeros(self.batch_size, self.frame_step).to(input.device)),
                1,
            )
        frames = input.unfold(1, self.frame_size, self.frame_step)

        out = torch.matmul(frames, self.mtx1)
        out = torch.pow(out, 2)
        out = torch.matmul(out, self.mtx2)
        out = out + 1e-10
        out = torch.log(out)

        out = self.linear1(out)
        out = F.relu(out)
        out = out.transpose(0, 1)
        out_full, (out, c_n) = self.lstm(out)
        out_full = out_full.transpose(0, 1)
        out = out[-1]
        out = self.linear2(out)

        return out


class SpeechClassifierDP(SpeechClassifier):
    def __init__(self, *args, **kwargs):
        super(SpeechClassifierDP, self).__init__(*args, **kwargs)

    def set_bound_method(self, bound_method):
        self.bound_method = bound_method

    def certify(self, input, gt, interim=None, max_iter=100, verbose=False):
        dev = input.device
        input.lb = torch.cat((input.lb, torch.zeros(self.frame_step).to(dev)), 0)
        input.ub = torch.cat((input.ub, torch.zeros(self.frame_step).to(dev)), 0)

        layers = []
        feed = []
        lstm_pack = []
        for frame_idx, i in enumerate(range(0, input.lb.size()[0], self.frame_step)):
            if i + self.frame_size >= input.lb.size()[0]:
                break
            if verbose:
                print(f"{frame_idx}-th frame")
            dpframe = R2.DeepPoly(
                input.lb[i : i + self.frame_size],
                input.ub[i : i + self.frame_size],
                None,
                None,
            )
            feed.append(dpframe)
            assert (dpframe.lb <= dpframe.ub).all(), "soundness check failed"

            chain = []
            # Speech pre-processing stage
            pre_linear1 = R2.Linear(self.frame_size, self.frame_size + 2)
            pre_linear1.assign(self.mtx1, device=dev)
            pl1_out = pre_linear1(dpframe)
            chain.append(pre_linear1)
            assert (interim is None) or (
                (pl1_out.lb <= interim[0][frame_idx]).all()
                and (interim[0][frame_idx] <= pl1_out.ub).all()
            ), "soundness check failed"

            pre_square = R2.Square(prev_layer=pre_linear1)
            psq_out = pre_square(pl1_out)
            chain.append(pre_square)
            assert (interim is None) or (
                (psq_out.lb <= interim[1][frame_idx]).all()
                and (interim[1][frame_idx] <= psq_out.ub).all()
            ), "soundness check failed"

            pre_linear2 = R2.Linear(
                self.frame_size + 2, self.n_filt, prev_layer=pre_square
            )
            pre_linear2.assign(self.mtx2, torch.ones(self.n_filt) * 1e-10, device=dev)
            pl2_out = pre_linear2(psq_out)
            chain.append(pre_linear2)
            assert (interim is None) or (
                (pl2_out.lb <= interim[2][frame_idx]).all()
                and (interim[2][frame_idx] <= pl2_out.ub).all()
            ), "soundness check failed"

            pre_log = R2.Log(prev_layer=pre_linear2)
            plg_out = pre_log(pl2_out)
            chain.append(pre_log)
            assert (interim is None) or (
                (plg_out.lb <= interim[3][frame_idx]).all()
                and (interim[3][frame_idx] <= plg_out.ub).all()
            ), "soundness check failed"

            # Neural network stage
            nn_linear1 = R2.Linear.convert(self.linear1, prev_layer=pre_log, device=dev)
            nl1_out = nn_linear1(plg_out)
            chain.append(nn_linear1)
            assert (interim is None) or (
                (nl1_out.lb <= interim[4][frame_idx]).all()
                and (interim[4][frame_idx] <= nl1_out.ub).all()
            ), "soundness check failed"

            nn_relu1 = R2.ReLU(prev_layer=nn_linear1).to(device=input.device)
            nr1_out = nn_relu1(nl1_out)
            chain.append(nn_relu1)
            assert (interim is None) or (
                (nr1_out.lb <= interim[5][frame_idx]).all()
                and (interim[5][frame_idx] <= nr1_out.ub).all()
            ), "soundness check failed"

            nn_lstm = R2.LSTMCell.convert(
                self.lstm,
                prev_layer=nn_relu1,
                prev_cell=None if frame_idx == 0 else lstm_pack[-1],
                method=self.bound_method,
                device=dev,
            )
            lstm_pack.append(nn_lstm)
            lstm_out = nn_lstm(nr1_out)
            chain.append(nn_lstm)
            if not (
                (interim is None)
                or (
                    (lstm_out.lb <= interim[6][frame_idx]).all()
                    and (interim[6][frame_idx] <= lstm_out.ub).all()
                )
            ):
                print(lstm_out.lb)
                print(interim[6][frame_idx])
                print(lstm_out.ub)
                print("max lb diff:", (interim[6][frame_idx] - lstm_out.lb).max())
                print("max ub diff:", (-interim[6][frame_idx] + lstm_out.ub).max())
                exit(0)

            layers.append(chain)

        nn_linear2 = R2.Linear.convert(
            self.linear2, prev_layer=lstm_pack[-1], device=dev
        )
        nl2_out = nn_linear2(lstm_out)
        layers.append(nn_linear2)
        assert (interim is None) or (
            (nl2_out.lb <= interim[7]).all() and (interim[7] <= nl2_out.ub).all()
        ), "soundness check failed"

        out_dim = nn_linear2.out_features
        lin_compare = R2.Linear(out_dim, 1, prev_layer=nn_linear2)
        layers.append(lin_compare)

        lp_proven = True
        for fl in range(out_dim):
            if fl == gt:
                continue
            if verbose:
                print(f"Testing label {fl} | ground truth {gt}")

            comp_mtx = torch.zeros(out_dim, 1)
            comp_mtx[gt, 0] = 1
            comp_mtx[fl, 0] = -1
            lin_compare.assign(comp_mtx, device=dev)
            lp_res = lin_compare(nl2_out)

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
