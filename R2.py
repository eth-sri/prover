import numpy as np
import math, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from relaxation import bounds as LP_bounds
from relaxation import ibp_bounds as IBP_bounds
from relaxation import LB_split, UB_split
from tqdm import tqdm


class DeepPoly:
    def __init__(self, lb, ub, lexpr, uexpr, device=None):
        self.lb = lb
        self.ub = ub
        self.lexpr = lexpr
        self.uexpr = uexpr
        assert not torch.isnan(self.lb).any()
        assert not torch.isnan(self.ub).any()
        assert lexpr is None or (
            (not torch.isnan(self.lexpr[0]).any())
            and (not torch.isnan(self.lexpr[1]).any())
        )
        assert uexpr is None or (
            (not torch.isnan(self.uexpr[0]).any())
            and (not torch.isnan(self.uexpr[1]).any())
        )
        self.dim = lb.size()[0]
        self.device = self.lb.device if device is None else device

    @staticmethod
    def deeppoly_from_perturbation(x, eps, truncate=None):
        assert eps >= 0, "epsilon must not be negative value"
        if truncate is not None:
            lb = x - eps
            ub = x + eps
            lb[lb < truncate[0]] = truncate[0]
            ub[ub > truncate[1]] = truncate[1]
            return DeepPoly(lb, ub, None, None)

        else:
            return DeepPoly(x - eps, x + eps, None, None)

    @staticmethod
    def deeppoly_from_dB_perturbation(x, eps_db):
        dBx = torch.log10(torch.abs(x).max()) * 20.0
        dBd = eps_db + dBx
        delta = 10 ** (dBd.item() / 20.0)
        return DeepPoly.deeppoly_from_perturbation(x, delta)


class DPBackSubstitution:
    def _get_lb(self, expr_w, expr_b):
        if len(self.output_dp.lexpr[0].size()) == 2:
            res_w = (
                positive_only(expr_w) @ self.output_dp.lexpr[0]
                + negative_only(expr_w) @ self.output_dp.uexpr[0]
            )
        else:
            res_w = (
                positive_only(expr_w) * self.output_dp.lexpr[0]
                + negative_only(expr_w) * self.output_dp.uexpr[0]
            )
        res_b = (
            positive_only(expr_w) @ self.output_dp.lexpr[1]
            + negative_only(expr_w) @ self.output_dp.uexpr[1]
            + expr_b
        )

        if self.prev_layer == None:
            return (
                positive_only(res_w) @ self.input_dp.lb
                + negative_only(res_w) @ self.input_dp.ub
                + res_b
            )
        else:
            return self.prev_layer._get_lb(res_w, res_b)

    def _get_ub(self, expr_w, expr_b):
        if len(self.output_dp.lexpr[0].size()) == 2:
            res_w = (
                positive_only(expr_w) @ self.output_dp.uexpr[0]
                + negative_only(expr_w) @ self.output_dp.lexpr[0]
            )
        else:
            res_w = (
                positive_only(expr_w) * self.output_dp.uexpr[0]
                + negative_only(expr_w) * self.output_dp.lexpr[0]
            )
        res_b = (
            positive_only(expr_w) @ self.output_dp.uexpr[1]
            + negative_only(expr_w) @ self.output_dp.lexpr[1]
            + expr_b
        )

        if self.prev_layer == None:
            return (
                positive_only(res_w) @ self.input_dp.ub
                + negative_only(res_w) @ self.input_dp.lb
                + res_b
            )
        else:
            return self.prev_layer._get_ub(res_w, res_b)


class LSTMCell(nn.LSTM, DPBackSubstitution):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        prev_layer=None,
        prev_cell=None,
        method="opt",
        device=torch.device("cpu"),
    ):
        super(LSTMCell, self).__init__(input_size, hidden_size)
        self.num_layers = num_layers
        self.prev_layer = prev_layer
        self.prev_cell = prev_cell
        self.h_t_dp = None

        self.matrices = []
        self.initialised = False
        if method in ["opt", "lp", "ibp"]:
            self.method = method
            if method == "opt":
                self.set_lambda(device)
        else:
            raise RuntimeError(f"not supported bounding method: {method}")

    @staticmethod
    def convert(
        cell, prev_layer=None, prev_cell=None, method="opt", device=torch.device("cpu")
    ):
        lstm = LSTMCell(
            cell.input_size,
            cell.hidden_size,
            cell.num_layers,
            prev_layer,
            prev_cell,
            method,
            device,
        )
        for idx in range(cell.num_layers):
            data_map = {}
            w_ii, w_if, w_ig, w_io = torch.split(
                getattr(cell, f"weight_ih_l{idx}"), cell.hidden_size, 0
            )
            w_hi, w_hf, w_hg, w_ho = torch.split(
                getattr(cell, f"weight_hh_l{idx}"), cell.hidden_size, 0
            )
            b_ii, b_if, b_ig, b_io = torch.split(
                getattr(cell, f"bias_ih_l{idx}"), cell.hidden_size, 0
            )
            b_hi, b_hf, b_hg, b_ho = torch.split(
                getattr(cell, f"bias_hh_l{idx}"), cell.hidden_size, 0
            )

            data_map["w_ii"] = w_ii
            data_map["w_if"] = w_if
            data_map["w_ig"] = w_ig
            data_map["w_io"] = w_io

            data_map["w_hi"] = w_hi
            data_map["w_hf"] = w_hf
            data_map["w_hg"] = w_hg
            data_map["w_ho"] = w_ho

            data_map["b_ii"] = b_ii
            data_map["b_if"] = b_if
            data_map["b_ig"] = b_ig
            data_map["b_io"] = b_io

            data_map["b_hi"] = b_hi
            data_map["b_hf"] = b_hf
            data_map["b_hg"] = b_hg
            data_map["b_ho"] = b_ho

            lstm.matrices.append(data_map)

        return lstm

    def set_lambda(self, device=torch.device("cpu")):
        self.lmb = (
            torch.Tensor(2, self.num_layers, 3, self.hidden_size, 5)
            .to(device)
            .uniform_(-1, 1)
        )
        self.lmb.requires_grad_()
        return self.lmb

    def _get_lb_bfs(self, layer_i, init_w_i, init_w_h, init_w_c, b):
        hsize = self.hidden_size
        isize = self.input_size
        dev = init_w_i.device

        cell = self
        frame_j = 0
        while cell.prev_cell != None:
            cell = cell.prev_cell
            frame_j += 1

        weight_map = {}
        bias = b.clone()
        for i in range(layer_i, -2, -1):
            for j in range(frame_j, -2, -1):
                jsize = isize if i == -1 else hsize
                weight_map[(i, j, "h")] = torch.zeros(init_w_h.size()[0], jsize).to(dev)
                weight_map[(i, j, "c")] = torch.zeros(init_w_h.size()[0], hsize).to(dev)
        weight_map[(layer_i - 1, frame_j, "h")] = init_w_i
        weight_map[(layer_i, frame_j - 1, "h")] = init_w_h
        weight_map[(layer_i, frame_j - 1, "c")] = init_w_c

        cell = self
        for j in range(frame_j, -1, -1):
            for i in range(layer_i, -1, -1):
                if i == layer_i and j == frame_j:
                    continue
                jsize = isize if i == 0 else hsize
                sz = [jsize, hsize, hsize]
                # so here we have to fill and update the weight map and the bias
                # update from (i, j, 'h')
                l_w_i, l_w_h, l_w_c = torch.split(cell.h_t_lexpr[i][0], sz, -1)
                u_w_i, u_w_h, u_w_c = torch.split(cell.h_t_uexpr[i][0], sz, -1)
                l_b, u_b = cell.h_t_lexpr[i][1], cell.h_t_uexpr[i][1]

                weight_map[(i - 1, j, "h")] += (
                    positive_only(weight_map[(i, j, "h")]) @ l_w_i
                    + negative_only(weight_map[(i, j, "h")]) @ u_w_i
                )
                weight_map[(i, j - 1, "h")] += (
                    positive_only(weight_map[(i, j, "h")]) @ l_w_h
                    + negative_only(weight_map[(i, j, "h")]) @ u_w_h
                )
                weight_map[(i, j - 1, "c")] += (
                    positive_only(weight_map[(i, j, "h")]) @ l_w_c
                    + negative_only(weight_map[(i, j, "h")]) @ u_w_c
                )
                bias += (
                    positive_only(weight_map[(i, j, "h")]) @ l_b
                    + negative_only(weight_map[(i, j, "h")]) @ u_b
                )

                # update from (i, j, 'c')
                l_w_i, l_w_h, l_w_c = torch.split(cell.c_t_lexpr[i][0], sz, -1)
                u_w_i, u_w_h, u_w_c = torch.split(cell.c_t_uexpr[i][0], sz, -1)
                l_b, u_b = cell.c_t_lexpr[i][1], cell.c_t_uexpr[i][1]

                weight_map[(i - 1, j, "h")] += (
                    positive_only(weight_map[(i, j, "c")]) @ l_w_i
                    + negative_only(weight_map[(i, j, "c")]) @ u_w_i
                )
                weight_map[(i, j - 1, "h")] += (
                    positive_only(weight_map[(i, j, "c")]) @ l_w_h
                    + negative_only(weight_map[(i, j, "c")]) @ u_w_h
                )
                weight_map[(i, j - 1, "c")] += (
                    positive_only(weight_map[(i, j, "c")]) @ l_w_c
                    + negative_only(weight_map[(i, j, "c")]) @ u_w_c
                )
                bias += (
                    positive_only(weight_map[(i, j, "c")]) @ l_b
                    + negative_only(weight_map[(i, j, "c")]) @ u_b
                )

            cell = cell.prev_cell

        res = torch.zeros(bias.size(), device=dev)
        cell = self
        for j in range(frame_j, -1, -1):
            res += cell.prev_layer._get_lb(
                weight_map[(-1, j, "h")], torch.zeros(bias.size()).to(dev)
            )
            cell = cell.prev_cell

        return res + bias

    def _get_lb(self, expr_w, expr_b):
        tmp_w = (
            positive_only(expr_w) @ self.h_t_dp.lexpr[0]
            + negative_only(expr_w) @ self.h_t_dp.uexpr[0]
        )
        tmp_b = (
            positive_only(expr_w) @ self.h_t_dp.lexpr[1]
            + negative_only(expr_w) @ self.h_t_dp.uexpr[1]
            + expr_b
        )
        expr_w = tmp_w
        expr_b = tmp_b

        hsize = self.hidden_size
        isize = self.input_size
        jsize = isize if self.num_layers == 1 else hsize
        sz = [jsize, hsize, hsize]
        w_i, w_h, w_c = torch.split(expr_w, sz, -1)
        return self._get_lb_bfs(self.num_layers - 1, w_i, w_h, w_c, expr_b)

    def _get_ub_bfs(self, layer_i, init_w_i, init_w_h, init_w_c, b):
        hsize = self.hidden_size
        isize = self.input_size
        dev = init_w_i.device

        cell = self
        frame_j = 0
        while cell.prev_cell != None:
            cell = cell.prev_cell
            frame_j += 1

        weight_map = {}
        bias = b.clone()
        for i in range(layer_i, -2, -1):
            for j in range(frame_j, -2, -1):
                jsize = isize if i == -1 else hsize
                weight_map[(i, j, "h")] = torch.zeros(init_w_h.size()[0], jsize).to(dev)
                weight_map[(i, j, "c")] = torch.zeros(init_w_h.size()[0], hsize).to(dev)
        weight_map[(layer_i - 1, frame_j, "h")] = init_w_i
        weight_map[(layer_i, frame_j - 1, "h")] = init_w_h
        weight_map[(layer_i, frame_j - 1, "c")] = init_w_c

        cell = self
        for j in range(frame_j, -1, -1):
            for i in range(layer_i, -1, -1):
                if i == layer_i and j == frame_j:
                    continue
                jsize = isize if i == 0 else hsize
                sz = [jsize, hsize, hsize]
                # so here we have to fill and update the weight map and the bias
                # update from (i, j, 'h')
                l_w_i, l_w_h, l_w_c = torch.split(cell.h_t_lexpr[i][0], sz, -1)
                u_w_i, u_w_h, u_w_c = torch.split(cell.h_t_uexpr[i][0], sz, -1)
                l_b, u_b = cell.h_t_lexpr[i][1], cell.h_t_uexpr[i][1]

                weight_map[(i - 1, j, "h")] += (
                    positive_only(weight_map[(i, j, "h")]) @ u_w_i
                    + negative_only(weight_map[(i, j, "h")]) @ l_w_i
                )
                weight_map[(i, j - 1, "h")] += (
                    positive_only(weight_map[(i, j, "h")]) @ u_w_h
                    + negative_only(weight_map[(i, j, "h")]) @ l_w_h
                )
                weight_map[(i, j - 1, "c")] += (
                    positive_only(weight_map[(i, j, "h")]) @ u_w_c
                    + negative_only(weight_map[(i, j, "h")]) @ l_w_c
                )
                bias += (
                    positive_only(weight_map[(i, j, "h")]) @ u_b
                    + negative_only(weight_map[(i, j, "h")]) @ l_b
                )

                # update from (i, j, 'c')
                l_w_i, l_w_h, l_w_c = torch.split(cell.c_t_lexpr[i][0], sz, -1)
                u_w_i, u_w_h, u_w_c = torch.split(cell.c_t_uexpr[i][0], sz, -1)
                l_b, u_b = cell.c_t_lexpr[i][1], cell.c_t_uexpr[i][1]

                weight_map[(i - 1, j, "h")] += (
                    positive_only(weight_map[(i, j, "c")]) @ u_w_i
                    + negative_only(weight_map[(i, j, "c")]) @ l_w_i
                )
                weight_map[(i, j - 1, "h")] += (
                    positive_only(weight_map[(i, j, "c")]) @ u_w_h
                    + negative_only(weight_map[(i, j, "c")]) @ l_w_h
                )
                weight_map[(i, j - 1, "c")] += (
                    positive_only(weight_map[(i, j, "c")]) @ u_w_c
                    + negative_only(weight_map[(i, j, "c")]) @ l_w_c
                )
                bias += (
                    positive_only(weight_map[(i, j, "c")]) @ u_b
                    + negative_only(weight_map[(i, j, "c")]) @ l_b
                )

            cell = cell.prev_cell

        res = torch.zeros(bias.size(), device=dev)
        cell = self
        for j in range(frame_j, -1, -1):
            res += cell.prev_layer._get_ub(
                weight_map[(-1, j, "h")], torch.zeros(bias.size()).to(dev)
            )
            cell = cell.prev_cell

        return res + bias

    def _get_ub(self, expr_w, expr_b):
        tmp_w = (
            positive_only(expr_w) @ self.h_t_dp.uexpr[0]
            + negative_only(expr_w) @ self.h_t_dp.lexpr[0]
        )
        tmp_b = (
            positive_only(expr_w) @ self.h_t_dp.uexpr[1]
            + negative_only(expr_w) @ self.h_t_dp.lexpr[1]
            + expr_b
        )
        expr_w = tmp_w
        expr_b = tmp_b

        hsize = self.hidden_size
        isize = self.input_size
        jsize = isize if self.num_layers == 1 else hsize
        sz = [jsize, hsize, hsize]
        w_i, w_h, w_c = torch.split(expr_w, sz, -1)

        return self._get_ub_bfs(self.num_layers - 1, w_i, w_h, w_c, expr_b)

    def forward(self, x_t_dp):
        hsize = self.hidden_size
        isize = self.input_size
        dev = x_t_dp.device

        self.h_t_lexpr = []
        self.h_t_uexpr = []
        self.c_t_lexpr = []
        self.c_t_uexpr = []

        sf = nn.Softmax(dim=-1)
        split_type = lambda p: 10 * (p // 3 + 1) + (2 - p % 2)

        for i in range(self.num_layers):
            jsize = isize if i == 0 else hsize
            sz = [jsize, hsize, hsize]

            w_ii, w_hi = self.matrices[i]["w_ii"], self.matrices[i]["w_hi"]
            w_i = torch.cat((w_ii, w_hi, torch.zeros(hsize, hsize).to(dev)), 1)
            b_i = self.matrices[i]["b_ii"] + self.matrices[i]["b_hi"]
            if not self.initialised:
                self.i_lb = self._get_lb_bfs(
                    i, w_ii, w_hi, torch.zeros(hsize, hsize).to(dev), b_i
                )
                self.i_ub = self._get_ub_bfs(
                    i, w_ii, w_hi, torch.zeros(hsize, hsize).to(dev), b_i
                )

            w_if, w_hf = self.matrices[i]["w_if"], self.matrices[i]["w_hf"]
            w_f = torch.cat((w_if, w_hf, torch.zeros(hsize, hsize).to(dev)), 1)
            b_f = self.matrices[i]["b_if"] + self.matrices[i]["b_hf"]
            if not self.initialised:
                self.f_lb = self._get_lb_bfs(
                    i, w_if, w_hf, torch.zeros(hsize, hsize).to(dev), b_f
                )
                self.f_ub = self._get_ub_bfs(
                    i, w_if, w_hf, torch.zeros(hsize, hsize).to(dev), b_f
                )

            w_ig, w_hg = self.matrices[i]["w_ig"], self.matrices[i]["w_hg"]
            w_g = torch.cat((w_ig, w_hg, torch.zeros(hsize, hsize).to(dev)), 1)
            b_g = self.matrices[i]["b_ig"] + self.matrices[i]["b_hg"]
            if not self.initialised:
                self.g_lb = self._get_lb_bfs(
                    i, w_ig, w_hg, torch.zeros(hsize, hsize).to(dev), b_g
                )
                self.g_ub = self._get_ub_bfs(
                    i, w_ig, w_hg, torch.zeros(hsize, hsize).to(dev), b_g
                )

            w_io, w_ho = self.matrices[i]["w_io"], self.matrices[i]["w_ho"]
            w_o = torch.cat((w_io, w_ho, torch.zeros(hsize, hsize).to(dev)), 1)
            b_o = self.matrices[i]["b_io"] + self.matrices[i]["b_ho"]
            if not self.initialised:
                self.o_lb = self._get_lb_bfs(
                    i, w_io, w_ho, torch.zeros(hsize, hsize).to(dev), b_o
                )
                self.o_ub = self._get_ub_bfs(
                    i, w_io, w_ho, torch.zeros(hsize, hsize).to(dev), b_o
                )

            w_c = torch.cat(
                (
                    torch.zeros(hsize, jsize),
                    torch.zeros(hsize, hsize),
                    torch.eye(hsize),
                ),
                1,
            ).to(dev)
            b_c = torch.zeros(hsize).to(device=dev)
            if not self.initialised:
                self.c_tm1_lb = self._get_lb_bfs(
                    i,
                    torch.zeros(hsize, jsize).to(dev),
                    torch.zeros(hsize, hsize).to(dev),
                    torch.eye(hsize).to(dev),
                    torch.zeros(hsize).to(dev),
                )
                self.c_tm1_ub = self._get_ub_bfs(
                    i,
                    torch.zeros(hsize, jsize).to(dev),
                    torch.zeros(hsize, hsize).to(dev),
                    torch.eye(hsize).to(dev),
                    torch.zeros(hsize).to(dev),
                )

            # c_t update
            lexpr_w_c = torch.zeros(hsize, jsize + 2 * hsize).to(dev)
            lexpr_b_c = torch.zeros(hsize).to(dev)
            uexpr_w_c = torch.zeros(hsize, jsize + 2 * hsize).to(dev)
            uexpr_b_c = torch.zeros(hsize).to(dev)

            if not self.initialised:
                self.precal_bnd = [{}, {}, {}]

            if self.prev_cell is not None:
                if not self.initialised:
                    coeff = torch.zeros(6, hsize, 5 if self.method == "opt" else 1).to(
                        dev
                    )
                    for d in range(hsize):
                        if self.method == "opt":
                            for p in range(5):
                                if p == 0:
                                    (
                                        coeff[0, d, p],
                                        coeff[1, d, p],
                                        coeff[2, d, p],
                                        coeff[3, d, p],
                                        coeff[4, d, p],
                                        coeff[5, d, p],
                                    ) = LP_bounds(
                                        self.f_lb[d],
                                        self.f_ub[d],
                                        self.c_tm1_lb[d],
                                        self.c_tm1_ub[d],
                                        tanh=False,
                                    )
                                else:
                                    (
                                        coeff[0, d, p],
                                        coeff[1, d, p],
                                        coeff[2, d, p],
                                        _,
                                    ) = LB_split(
                                        self.f_lb[d].item(),
                                        self.f_ub[d].item(),
                                        self.c_tm1_lb[d].item(),
                                        self.c_tm1_ub[d].item(),
                                        tanh=False,
                                        split_type=split_type(p),
                                    )
                                    (
                                        coeff[3, d, p],
                                        coeff[4, d, p],
                                        coeff[5, d, p],
                                        _,
                                    ) = UB_split(
                                        self.f_lb[d].item(),
                                        self.f_ub[d].item(),
                                        self.c_tm1_lb[d].item(),
                                        self.c_tm1_ub[d].item(),
                                        tanh=False,
                                        split_type=split_type(p),
                                    )
                        elif self.method == "lp":
                            (
                                coeff[0, d, 0],
                                coeff[1, d, 0],
                                coeff[2, d, 0],
                                coeff[3, d, 0],
                                coeff[4, d, 0],
                                coeff[5, d, 0],
                            ) = LP_bounds(
                                self.f_lb[d],
                                self.f_ub[d],
                                self.c_tm1_lb[d],
                                self.c_tm1_ub[d],
                                tanh=False,
                            )
                        elif self.method == "ibp":
                            (
                                coeff[0, d, 0],
                                coeff[1, d, 0],
                                coeff[2, d, 0],
                                coeff[3, d, 0],
                                coeff[4, d, 0],
                                coeff[5, d, 0],
                            ) = IBP_bounds(
                                self.f_lb[d],
                                self.f_ub[d],
                                self.c_tm1_lb[d],
                                self.c_tm1_ub[d],
                                tanh=False,
                            )

                    self.precal_bnd[0] = coeff
                    Al = coeff[0, :, 0]
                    Bl = coeff[1, :, 0]
                    Cl = coeff[2, :, 0]
                    Au = coeff[3, :, 0]
                    Bu = coeff[4, :, 0]
                    Cu = coeff[5, :, 0]

                else:
                    if self.method == "opt":
                        lmb_l = sf(self.lmb[0, i, 0])
                        coeff = self.precal_bnd[0][:3] * lmb_l
                        Al = torch.sum(coeff[0], -1)
                        Bl = torch.sum(coeff[1], -1)
                        Cl = torch.sum(coeff[2], -1)
                        lmb_u = sf(self.lmb[1, i, 0])
                        coeff = self.precal_bnd[0][3:] * lmb_u
                        Au = torch.sum(coeff[0], -1)
                        Bu = torch.sum(coeff[1], -1)
                        Cu = torch.sum(coeff[2], -1)
                    else:
                        Al = self.precal_bnd[0][0, :, 0]
                        Bl = self.precal_bnd[0][1, :, 0]
                        Cl = self.precal_bnd[0][2, :, 0]
                        Au = self.precal_bnd[0][3, :, 0]
                        Bu = self.precal_bnd[0][4, :, 0]
                        Cu = self.precal_bnd[0][5, :, 0]

                lexpr_w_c += torch.diag(Al) @ w_f + torch.diag(Bl) @ w_c
                lexpr_b_c += torch.diag(Al) @ b_f + torch.diag(Bl) @ b_c + Cl
                uexpr_w_c += torch.diag(Au) @ w_f + torch.diag(Bu) @ w_c
                uexpr_b_c += torch.diag(Au) @ b_f + torch.diag(Bu) @ b_c + Cu

            if not self.initialised:
                coeff = torch.zeros(6, hsize, 5 if self.method == "opt" else 1).to(dev)
                for d in range(hsize):
                    if self.method == "opt":
                        for p in range(5):
                            if p == 0:
                                (
                                    coeff[0, d, p],
                                    coeff[1, d, p],
                                    coeff[2, d, p],
                                    coeff[3, d, p],
                                    coeff[4, d, p],
                                    coeff[5, d, p],
                                ) = LP_bounds(
                                    self.i_lb[d],
                                    self.i_ub[d],
                                    self.g_lb[d],
                                    self.g_ub[d],
                                    tanh=True,
                                )
                            else:
                                (
                                    coeff[0, d, p],
                                    coeff[1, d, p],
                                    coeff[2, d, p],
                                    _,
                                ) = LB_split(
                                    self.i_lb[d].item(),
                                    self.i_ub[d].item(),
                                    self.g_lb[d].item(),
                                    self.g_ub[d].item(),
                                    tanh=True,
                                    split_type=split_type(p),
                                )
                                (
                                    coeff[3, d, p],
                                    coeff[4, d, p],
                                    coeff[5, d, p],
                                    _,
                                ) = UB_split(
                                    self.i_lb[d].item(),
                                    self.i_ub[d].item(),
                                    self.g_lb[d].item(),
                                    self.g_ub[d].item(),
                                    tanh=True,
                                    split_type=split_type(p),
                                )
                    elif self.method == "lp":
                        (
                            coeff[0, d, 0],
                            coeff[1, d, 0],
                            coeff[2, d, 0],
                            coeff[3, d, 0],
                            coeff[4, d, 0],
                            coeff[5, d, 0],
                        ) = LP_bounds(
                            self.i_lb[d],
                            self.i_ub[d],
                            self.g_lb[d],
                            self.g_ub[d],
                            tanh=True,
                        )
                    elif self.method == "ibp":
                        (
                            coeff[0, d, 0],
                            coeff[1, d, 0],
                            coeff[2, d, 0],
                            coeff[3, d, 0],
                            coeff[4, d, 0],
                            coeff[5, d, 0],
                        ) = IBP_bounds(
                            self.i_lb[d],
                            self.i_ub[d],
                            self.g_lb[d],
                            self.g_ub[d],
                            tanh=True,
                        )

                self.precal_bnd[1] = coeff
                Al = coeff[0, :, 0]
                Bl = coeff[1, :, 0]
                Cl = coeff[2, :, 0]
                Au = coeff[3, :, 0]
                Bu = coeff[4, :, 0]
                Cu = coeff[5, :, 0]

            else:
                if self.method == "opt":
                    lmb_l = sf(self.lmb[0, i, 1])
                    coeff = self.precal_bnd[1][:3] * lmb_l
                    Al = torch.sum(coeff[0], -1)
                    Bl = torch.sum(coeff[1], -1)
                    Cl = torch.sum(coeff[2], -1)
                    lmb_u = sf(self.lmb[1, i, 1])
                    coeff = self.precal_bnd[1][3:] * lmb_u
                    Au = torch.sum(coeff[0], -1)
                    Bu = torch.sum(coeff[1], -1)
                    Cu = torch.sum(coeff[2], -1)
                else:
                    Al = self.precal_bnd[1][0, :, 0]
                    Bl = self.precal_bnd[1][1, :, 0]
                    Cl = self.precal_bnd[1][2, :, 0]
                    Au = self.precal_bnd[1][3, :, 0]
                    Bu = self.precal_bnd[1][4, :, 0]
                    Cu = self.precal_bnd[1][5, :, 0]

            lexpr_w_c += torch.diag(Al) @ w_i + torch.diag(Bl) @ w_g
            lexpr_b_c += torch.diag(Al) @ b_i + torch.diag(Bl) @ b_g + Cl
            uexpr_w_c += torch.diag(Au) @ w_i + torch.diag(Bu) @ w_g
            uexpr_b_c += torch.diag(Au) @ b_i + torch.diag(Bu) @ b_g + Cu

            self.c_t_lexpr.append((lexpr_w_c, lexpr_b_c))
            self.c_t_uexpr.append((uexpr_w_c, uexpr_b_c))

            if not self.initialised:
                w1, w2, w3 = torch.split(lexpr_w_c, sz, -1)
                w4, w5, w6 = torch.split(uexpr_w_c, sz, -1)
                self.c_t_lb = self._get_lb_bfs(i, w1, w2, w3, lexpr_b_c)
                self.c_t_ub = self._get_ub_bfs(i, w4, w5, w6, uexpr_b_c)

            # h_t update
            lexpr_w_h = torch.zeros(hsize, jsize + 2 * hsize, device=dev)
            lexpr_b_h = torch.zeros(hsize, device=dev)
            uexpr_w_h = torch.zeros(hsize, jsize + 2 * hsize, device=dev)
            uexpr_b_h = torch.zeros(hsize, device=dev)

            if not self.initialised:
                coeff = torch.zeros(6, hsize, 5 if self.method == "opt" else 1).to(dev)
                for d in range(hsize):
                    if self.method == "opt":
                        for p in range(5):
                            if p == 0:
                                (
                                    coeff[0, d, p],
                                    coeff[1, d, p],
                                    coeff[2, d, p],
                                    coeff[3, d, p],
                                    coeff[4, d, p],
                                    coeff[5, d, p],
                                ) = LP_bounds(
                                    self.o_lb[d],
                                    self.o_ub[d],
                                    self.c_t_lb[d],
                                    self.c_t_ub[d],
                                    tanh=True,
                                )
                            else:
                                (
                                    coeff[0, d, p],
                                    coeff[1, d, p],
                                    coeff[2, d, p],
                                    _,
                                ) = LB_split(
                                    self.o_lb[d].item(),
                                    self.o_ub[d].item(),
                                    self.c_t_lb[d].item(),
                                    self.c_t_ub[d].item(),
                                    tanh=True,
                                    split_type=split_type(p),
                                )
                                (
                                    coeff[3, d, p],
                                    coeff[4, d, p],
                                    coeff[5, d, p],
                                    _,
                                ) = UB_split(
                                    self.o_lb[d].item(),
                                    self.o_ub[d].item(),
                                    self.c_t_lb[d].item(),
                                    self.c_t_ub[d].item(),
                                    tanh=True,
                                    split_type=split_type(p),
                                )
                    elif self.method == "lp":
                        (
                            coeff[0, d, 0],
                            coeff[1, d, 0],
                            coeff[2, d, 0],
                            coeff[3, d, 0],
                            coeff[4, d, 0],
                            coeff[5, d, 0],
                        ) = LP_bounds(
                            self.o_lb[d],
                            self.o_ub[d],
                            self.c_t_lb[d],
                            self.c_t_ub[d],
                            tanh=True,
                        )
                    elif self.method == "ibp":
                        (
                            coeff[0, d, 0],
                            coeff[1, d, 0],
                            coeff[2, d, 0],
                            coeff[3, d, 0],
                            coeff[4, d, 0],
                            coeff[5, d, 0],
                        ) = IBP_bounds(
                            self.o_lb[d],
                            self.o_ub[d],
                            self.c_t_lb[d],
                            self.c_t_ub[d],
                            tanh=True,
                        )

                self.precal_bnd[2] = coeff
                Al = coeff[0, :, 0]
                Bl = coeff[1, :, 0]
                Cl = coeff[2, :, 0]
                Au = coeff[3, :, 0]
                Bu = coeff[4, :, 0]
                Cu = coeff[5, :, 0]

            else:
                if self.method == "opt":
                    lmb_l = sf(self.lmb[0, i, 2])
                    coeff = self.precal_bnd[2][:3] * lmb_l
                    Al = torch.sum(coeff[0], -1)
                    Bl = torch.sum(coeff[1], -1)
                    Cl = torch.sum(coeff[2], -1)
                    lmb_u = sf(self.lmb[1, i, 2])
                    coeff = self.precal_bnd[2][3:] * lmb_u
                    Au = torch.sum(coeff[0], -1)
                    Bu = torch.sum(coeff[1], -1)
                    Cu = torch.sum(coeff[2], -1)
                else:
                    Al = self.precal_bnd[2][0, :, 0]
                    Bl = self.precal_bnd[2][1, :, 0]
                    Cl = self.precal_bnd[2][2, :, 0]
                    Au = self.precal_bnd[2][3, :, 0]
                    Bu = self.precal_bnd[2][4, :, 0]
                    Cu = self.precal_bnd[2][5, :, 0]

            lexpr_w_h = (
                torch.diag(Al) @ w_o
                + torch.diag(positive_only(Bl)) @ lexpr_w_c
                + torch.diag(negative_only(Bl)) @ uexpr_w_c
            )
            lexpr_b_h = (
                torch.diag(Al) @ b_o
                + torch.diag(positive_only(Bl)) @ lexpr_b_c
                + torch.diag(negative_only(Bl)) @ uexpr_b_c
                + Cl
            )
            uexpr_w_h = (
                torch.diag(Au) @ w_o
                + torch.diag(positive_only(Bu)) @ uexpr_w_c
                + torch.diag(negative_only(Bu)) @ lexpr_w_c
            )
            uexpr_b_h = (
                torch.diag(Au) @ b_o
                + torch.diag(positive_only(Bu)) @ uexpr_b_c
                + torch.diag(negative_only(Bu)) @ lexpr_b_c
                + Cu
            )

            self.h_t_lexpr.append((lexpr_w_h, lexpr_b_h))
            self.h_t_uexpr.append((uexpr_w_h, uexpr_b_h))

            if not self.initialised:
                w1, w2, w3 = torch.split(lexpr_w_h, sz, -1)
                w4, w5, w6 = torch.split(uexpr_w_h, sz, -1)
                self.h_t_lb = self._get_lb_bfs(i, w1, w2, w3, lexpr_b_h)
                self.h_t_ub = self._get_ub_bfs(i, w4, w5, w6, uexpr_b_h)

        self.h_t_dp = DeepPoly(
            lb=self.h_t_lb,
            ub=self.h_t_ub,
            lexpr=(lexpr_w_h, lexpr_b_h),
            uexpr=(uexpr_w_h, uexpr_b_h),
            device=dev,
        )
        self.initialised = True

        return self.h_t_dp


class Linear(nn.Linear, DPBackSubstitution):
    def __init__(self, in_features, out_features, bias=True, prev_layer=None):
        super(Linear, self).__init__(in_features, out_features, bias)
        self.prev_layer = prev_layer
        self.input_dp = None
        self.output_dp = None

    @staticmethod
    def convert(layer, prev_layer=None, device=torch.device("cpu")):
        l = Linear(
            layer.in_features, layer.out_features, layer.bias is not None, prev_layer
        )
        l.weight.data = layer.weight.data.to(device)
        l.bias.data = layer.bias.data.to(device)
        return l

    def assign(self, weight, bias=None, device=torch.device("cpu")):
        assert weight.size() == torch.Size([self.in_features, self.out_features])
        assert bias is None or bias.size() == torch.Size([self.out_features])
        self.weight.data = weight.data.to(device).t()
        if bias is not None:
            self.bias.data = bias.data.to(device)
        else:
            self.bias.data = torch.zeros(self.out_features).to(device)

    def forward(self, prev_dp):
        # Initial layer
        if self.prev_layer == None:
            self.input_dp = prev_dp
            lb = (
                positive_only(self.weight) @ self.input_dp.lb
                + negative_only(self.weight) @ self.input_dp.ub
                + self.bias
            )
            ub = (
                positive_only(self.weight) @ self.input_dp.ub
                + negative_only(self.weight) @ self.input_dp.lb
                + self.bias
            )

        # Intermediate layer
        else:
            lb = self.prev_layer._get_lb(self.weight, self.bias)
            ub = self.prev_layer._get_ub(self.weight, self.bias)

        self.output_dp = DeepPoly(
            lb=lb,
            ub=ub,
            lexpr=(self.weight, self.bias),
            uexpr=(self.weight, self.bias),
            device=prev_dp.device,
        )

        return self.output_dp


class Square(nn.ReLU, DPBackSubstitution):
    def __init__(self, prev_layer=None):
        super(Square, self).__init__()
        self.prev_layer = prev_layer
        self.output_dp = None

    def forward(self, prev_dp):
        dim = prev_dp.dim
        dev = prev_dp.device
        lexpr_w = torch.zeros(dim).to(device=dev)
        lexpr_b = torch.zeros(dim).to(device=dev)
        uexpr_w = torch.zeros(dim).to(device=dev)
        uexpr_b = torch.zeros(dim).to(device=dev)
        ty = 1e-5
        tx = math.sqrt(ty)

        # intermediate layer
        for i in range(dim):
            l, u = prev_dp.lb[i], prev_dp.ub[i]
            uexpr_w[i] = l + u
            uexpr_b[i] = -l * u
            if tx <= l and 3 * l * l + 2 * u * l - u * u <= 4 * ty:
                p = l + math.sqrt(l * l - ty)
                lexpr_w[i] = 2 * p
                lexpr_b[i] = -p * p
            elif -tx >= u and 3 * u * u + 2 * u * l - l * l <= 4 * ty:
                p = u - math.sqrt(u * u - ty)
                lexpr_w[i] = 2 * p
                lexpr_b[i] = -p * p
            elif l <= tx and -tx <= u:
                pass
            else:
                lexpr_w[i] = l + u
                lexpr_b[i] = -(((l + u) / 2) ** 2)

        lb = self.prev_layer._get_lb(torch.diag(lexpr_w), lexpr_b)
        ub = self.prev_layer._get_ub(torch.diag(uexpr_w), uexpr_b)
        self.output_dp = DeepPoly(
            lb=lb,
            ub=ub,
            lexpr=(lexpr_w, lexpr_b),
            uexpr=(uexpr_w, uexpr_b),
            device=prev_dp.device,
        )

        return self.output_dp


class Log(nn.ReLU, DPBackSubstitution):
    def __init__(self, prev_layer=None):
        super(Log, self).__init__()
        self.prev_layer = prev_layer
        self.output_dp = None

    def forward(self, prev_dp):
        dim = prev_dp.dim
        dev = prev_dp.device
        lexpr_w = torch.zeros(dim).to(device=dev)
        lexpr_b = torch.zeros(dim).to(device=dev)
        uexpr_w = torch.zeros(dim).to(device=dev)
        uexpr_b = torch.zeros(dim).to(device=dev)
        trick = torch.Tensor([1e-10]).to(device=dev)  # to avoid log(0)

        # intermediate layer
        for i in range(dim):
            l = torch.max(prev_dp.lb[i], trick)
            u = torch.max(l + trick, prev_dp.ub[i])
            try:
                lexpr_w[i] = math.log(u / l) / (u - l)
                lexpr_b[i] = math.log(l) - math.log(u / l) * l / (u - l)
                uexpr_w[i] = 2 / (l + u)
                uexpr_b[i] = math.log((l + u) / 2) - 1
            except:
                print(f"ValueError caught: at {i}, l: {l}, u: {u}")
                print(prev_dp.lb, prev_dp.ub)
                exit(0)

        lb = self.prev_layer._get_lb(torch.diag(lexpr_w), lexpr_b)
        ub = self.prev_layer._get_ub(torch.diag(uexpr_w), uexpr_b)
        if torch.isnan(lb).any() or torch.isnan(ub).any():
            print(prev_dp.lb)
            print(prev_dp.ub)
            print(lb)
            print(ub)
            exit(0)
        self.output_dp = DeepPoly(
            lb=lb,
            ub=ub,
            lexpr=(lexpr_w, lexpr_b),
            uexpr=(uexpr_w, uexpr_b),
            device=prev_dp.device,
        )

        return self.output_dp


class ReLU(nn.ReLU, DPBackSubstitution):
    def __init__(self, inplace=False, prev_layer=None):
        super(ReLU, self).__init__(inplace)
        self.prev_layer = prev_layer
        self.output_dp = None

    def forward(self, prev_dp):
        dim = prev_dp.dim
        dev = prev_dp.device
        lexpr_w = torch.zeros(dim).to(device=dev)
        lexpr_b = torch.zeros(dim).to(device=dev)
        uexpr_w = torch.zeros(dim).to(device=dev)
        uexpr_b = torch.zeros(dim).to(device=dev)

        # intermediate layer
        for i in range(dim):
            l, u = prev_dp.lb[i], prev_dp.ub[i]
            if l > 0:
                lexpr_w[i] = 1
                uexpr_w[i] = 1
            elif u < 0:
                pass
            else:
                lexpr_w[i] = 1 if -l < u else 0
                uexpr_w[i] = u / (u - l)
                uexpr_b[i] = -l * u / (u - l)

        lb = self.prev_layer._get_lb(torch.diag(lexpr_w), lexpr_b)
        ub = self.prev_layer._get_ub(torch.diag(uexpr_w), uexpr_b)
        self.output_dp = DeepPoly(
            lb=lb,
            ub=ub,
            lexpr=(lexpr_w, lexpr_b),
            uexpr=(uexpr_w, uexpr_b),
            device=prev_dp.device,
        )

        return self.output_dp


class Sigmoidal(nn.Sigmoid, DPBackSubstitution):
    def __init__(self, func, prev_layer=None):
        if func in ["sigmoid", "tanh"]:
            super(Sigmoidal, self).__init__()
            self.func = func
        else:
            raise RuntimeError("not supported sigmoidal layer")
        self.prev_layer = prev_layer
        self.output_dp = None

    def forward(self, prev_dp):
        dim = prev_dp.dim
        dev = prev_dp.device
        lexpr_w = torch.zeros(dim).to(device=dev)
        lexpr_b = torch.zeros(dim).to(device=dev)
        uexpr_w = torch.zeros(dim).to(device=dev)
        uexpr_b = torch.zeros(dim).to(device=dev)

        # intermediate layer
        for i in range(dim):
            l, u = prev_dp.lb[i], prev_dp.ub[i]
            if self.func == "sigmoid":
                sl, su = torch.sigmoid(l), torch.sigmoid(u)
                lmb = (su - sl) / (u - l) if l < u else sl * (1 - sl)
                lmb_ = torch.min(su * (1 - su), sl * (1 - sl))
            elif self.func == "tanh":
                sl, su = torch.tanh(l), torch.tanh(u)
                lmb = (su - sl) / (u - l) if l < u else 1 - sl * sl
                lmb_ = torch.min(1 - su * su, 1 - sl * sl)
            if l > 0:
                lexpr_w[i] = lmb
                lexpr_b[i] = sl - lmb * l
            else:
                lexpr_w[i] = lmb_
                lexpr_b[i] = sl - lmb_ * l
            if u < 0:
                uexpr_w[i] = lmb
                uexpr_b[i] = su - lmb * u
            else:
                uexpr_w[i] = lmb_
                uexpr_b[i] = su - lmb_ * u

        lb = self.prev_layer._get_lb(torch.diag(lexpr_w), lexpr_b)
        ub = self.prev_layer._get_ub(torch.diag(uexpr_w), uexpr_b)
        self.output_dp = DeepPoly(
            lb=lb,
            ub=ub,
            lexpr=(lexpr_w, lexpr_b),
            uexpr=(uexpr_w, uexpr_b),
            device=prev_dp.device,
        )

        return self.output_dp
