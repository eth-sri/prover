from gurobipy import *
import numpy as np
import torch


def f(x, y, tanh):
    # Calculates sigmoid(x) * tanh(y) or sigmoid(x) * y.
    if tanh:
        return np.tanh(y) / (1 + np.exp(-x))
    else:
        return y / (1 + np.exp(-x))


def sigma(x):
    # Calculates sigmoid(x).
    return 1 / (1 + np.exp(-x))


def ibp_bounds(lx, ux, ly, uy, tanh=True):
    # Calculates IBP bounds of [lx,ux] X [ly,uy].
    if type(lx) is torch.Tensor:
        lx = lx.item()
    if type(ux) is torch.Tensor:
        ux = ux.item()
    if type(ly) is torch.Tensor:
        ly = ly.item()
    if type(uy) is torch.Tensor:
        uy = uy.item()
    candidates = torch.Tensor(
        [f(lx, ly, tanh), f(lx, uy, tanh), f(ux, ly, tanh), f(ux, uy, tanh)]
    )
    return 0, 0, torch.min(candidates), 0, 0, torch.max(candidates)


def proper_roots(equ, lx, ux, ly, uy, var="x", fn=(lambda v: v)):
    # Equation solver and filter the roots satisfying the proper conditions.
    Xs = np.zeros([0])
    Ys = np.zeros([0])
    roots = np.roots(equ)
    if var == "x":
        for sx in roots:
            if np.iscomplex(sx) or sx >= 1 or sx <= 0:
                continue
            sx = np.real(sx)
            x = -np.log((1 - sx) / sx)
            y = fn(sx)
            if lx <= x and x <= ux and ly <= y and y <= uy:
                Xs = np.concatenate([Xs, [x]])
                Ys = np.concatenate([Ys, [y]])
    elif var == "y":
        for ty in roots:
            if np.iscomplex(ty) or ty >= 1 or ty <= -1:
                continue
            ty = np.real(ty)
            y = np.arctanh(ty)
            x = fn(ty)
            if lx <= x and x <= ux and ly <= y and y <= uy:
                Xs = np.concatenate([Xs, [x]])
                Ys = np.concatenate([Ys, [y]])
    return Xs, Ys


def get_LB_delta(Al, Bl, Cl, lx, ux, ly, uy, tanh):
    # Cl calibration.
    bndX = np.array([lx, lx, ux, ux])
    bndY = np.array([ly, uy, ly, uy])
    if not tanh:
        # Case 2 only
        Xs1, Ys1 = (
            proper_roots([1, -1, Al / ly], lx, ux, ly, uy, var="x", fn=(lambda v: ly))
            if ly != 0
            else ([], [])
        )
        Xs2, Ys2 = (
            proper_roots([1, -1, Al / uy], lx, ux, ly, uy, var="x", fn=(lambda v: uy))
            if uy != 0
            else ([], [])
        )
        bndX = np.concatenate([bndX, Xs1, Xs2])
        bndY = np.concatenate([bndY, Ys1, Ys2])
    if tanh:
        # Case 1
        Xs1, Ys1 = proper_roots(
            [1, 0, Bl / sigma(lx) - 1], lx, ux, ly, uy, var="y", fn=(lambda v: lx)
        )
        Xs2, Ys2 = proper_roots(
            [1, 0, Bl / sigma(ux) - 1], lx, ux, ly, uy, var="y", fn=(lambda v: ux)
        )
        # Case 2
        Xs3, Ys3 = (
            proper_roots(
                [1, -1, Al / np.tanh(ly)], lx, ux, ly, uy, var="x", fn=(lambda v: ly)
            )
            if ly != 0
            else ([], [])
        )
        Xs4, Ys4 = (
            proper_roots(
                [1, -1, Al / np.tanh(uy)], lx, ux, ly, uy, var="x", fn=(lambda v: uy)
            )
            if uy != 0
            else ([], [])
        )
        # Case 3
        Xs5, Ys5 = proper_roots(
            [1, -2 - Bl, 1 + 2 * Bl, -Bl, -Al * Al],
            lx,
            ux,
            ly,
            uy,
            var="x",
            fn=(
                lambda v: np.arctanh(Al / v / (1 - v))
                if abs(Al / v / (1 - v)) < 1
                else uy + 1
            ),
        )
        bndX = np.concatenate([bndX, Xs1, Xs2, Xs3, Xs4, Xs5])
        bndY = np.concatenate([bndY, Ys1, Ys2, Ys3, Ys4, Ys5])

    delta = np.min(f(bndX, bndY, tanh) - Al * bndX - Bl * bndY - Cl)
    return delta


def get_UB_delta(Au, Bu, Cu, lx, ux, ly, uy, tanh):
    # Cu calibration.
    bndX = np.array([lx, lx, ux, ux])
    bndY = np.array([ly, uy, ly, uy])
    if not tanh:
        # Case 2 only
        Xs1, Ys1 = (
            proper_roots([1, -1, Au / ly], lx, ux, ly, uy, var="x", fn=(lambda v: ly))
            if ly != 0
            else ([], [])
        )
        Xs2, Ys2 = (
            proper_roots([1, -1, Au / uy], lx, ux, ly, uy, var="x", fn=(lambda v: uy))
            if uy != 0
            else ([], [])
        )
        bndX = np.concatenate([bndX, Xs1, Xs2])
        bndY = np.concatenate([bndY, Ys1, Ys2])
    if tanh:
        # Case 1
        Xs1, Ys1 = proper_roots(
            [1, 0, Bu / sigma(lx) - 1], lx, ux, ly, uy, var="y", fn=(lambda v: lx)
        )
        Xs2, Ys2 = proper_roots(
            [1, 0, Bu / sigma(ux) - 1], lx, ux, ly, uy, var="y", fn=(lambda v: ux)
        )
        # Case 2
        Xs3, Ys3 = (
            proper_roots(
                [1, -1, Au / np.tanh(ly)], lx, ux, ly, uy, var="x", fn=(lambda v: ly)
            )
            if ly != 0
            else ([], [])
        )
        Xs4, Ys4 = (
            proper_roots(
                [1, -1, Au / np.tanh(uy)], lx, ux, ly, uy, var="x", fn=(lambda v: uy)
            )
            if uy != 0
            else ([], [])
        )
        # Case 3
        Xs5, Ys5 = proper_roots(
            [1, -2 - Bu, 1 + 2 * Bu, -Bu, -Au * Au],
            lx,
            ux,
            ly,
            uy,
            var="x",
            fn=(
                lambda v: np.arctanh(Au / v / (1 - v))
                if abs(Au / v / (1 - v)) < 1
                else uy + 1
            ),
        )
        bndX = np.concatenate([bndX, Xs1, Xs2, Xs3, Xs4, Xs5])
        bndY = np.concatenate([bndY, Ys1, Ys2, Ys3, Ys4, Ys5])

    delta = np.min(Au * bndX + Bu * bndY + Cu - f(bndX, bndY, tanh))
    return delta


def LB(lx, ux, ly, uy, tanh=True, n_samples=100):
    # Calculate Al, Bl, Cl by sampling and linear programming.
    bndX = np.array([lx, lx, ux, ux])
    bndY = np.array([ly, uy, ly, uy])
    X = np.concatenate([bndX, np.random.uniform(lx, ux, n_samples - 4)])
    Y = np.concatenate([bndY, np.random.uniform(ly, uy, n_samples - 4)])

    model = Model()
    model.setParam("OutputFlag", 0)

    Al = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Al")
    Bl = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Bl")
    Cl = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Cl")

    model.addConstrs(
        (Al * X[i] + Bl * Y[i] + Cl <= f(X[i], Y[i], tanh) for i in range(n_samples)),
        name="ctr",
    )

    obj = LinExpr()
    obj = np.sum(f(X, Y, tanh)) - Al * np.sum(X) - Bl * np.sum(Y) - Cl * n_samples
    model.setObjective(obj, GRB.MINIMIZE)

    model.optimize()

    if model.status == GRB.Status.OPTIMAL:
        Al, Bl, Cl = model.getAttr("x", model.getVars())
        delta = get_LB_delta(Al, Bl, Cl, lx, ux, ly, uy, tanh)
        Cl += delta
        return Al, Bl, Cl
    else:
        return None, None, None


def UB(lx, ux, ly, uy, tanh=True, n_samples=100):
    # Calculate Au, Bu, Cu by sampling and linear programming.
    bndX = np.array([lx, lx, ux, ux])
    bndY = np.array([ly, uy, ly, uy])
    X = np.concatenate([bndX, np.random.uniform(lx, ux, n_samples - 4)])
    Y = np.concatenate([bndY, np.random.uniform(ly, uy, n_samples - 4)])

    model = Model()
    model.setParam("OutputFlag", 0)

    Au = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Au")
    Bu = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Bu")
    Cu = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Cu")

    model.addConstrs(
        (Au * X[i] + Bu * Y[i] + Cu >= f(X[i], Y[i], tanh) for i in range(n_samples)),
        name="ctr",
    )

    obj = LinExpr()
    obj = Au * np.sum(X) + Bu * np.sum(Y) + Cu * n_samples - np.sum(f(X, Y, tanh))
    model.setObjective(obj, GRB.MINIMIZE)

    model.optimize()

    if model.status == GRB.Status.OPTIMAL:
        Au, Bu, Cu = model.getAttr("x", model.getVars())
        delta = get_UB_delta(Au, Bu, Cu, lx, ux, ly, uy, tanh)
        Cu -= delta
        return Au, Bu, Cu
    else:
        return None, None, None


def bounds(lx, ux, ly, uy, tanh=True):
    # Caller function to obtain lower and upper bounding planes.
    if type(lx) is torch.Tensor:
        lx = lx.item()
    if type(ux) is torch.Tensor:
        ux = ux.item()
    if type(ly) is torch.Tensor:
        ly = ly.item()
    if type(uy) is torch.Tensor:
        uy = uy.item()
    Al, Bl, Cl = LB(lx, ux, ly, uy, tanh)
    Au, Bu, Cu = UB(lx, ux, ly, uy, tanh)
    return Al, Bl, Cl, Au, Bu, Cu


def LB_split(lx, ux, ly, uy, tanh=True, split_type=0, n_samples=200):
    # Get lower bound plane with triangular domain.
    X = np.random.uniform(lx, ux, n_samples)
    Y = np.random.uniform(ly, uy, n_samples)

    if split_type == 11:
        sel = (ux - lx) * (Y - ly) <= (uy - ly) * (X - lx)
    elif split_type == 12:
        sel = (ux - lx) * (Y - ly) >= (uy - ly) * (X - lx)
    elif split_type == 21:
        sel = (ux - lx) * (Y - ly) <= (ly - uy) * (X - ux)
    elif split_type == 22:
        sel = (ux - lx) * (Y - ly) >= (ly - uy) * (X - ux)
    X, Y = X[sel], Y[sel]
    bndX = np.array([lx, lx, ux, ux])
    bndY = np.array([ly, uy, ly, uy])
    X = np.concatenate([bndX, X])
    Y = np.concatenate([bndY, Y])
    n = X.shape[0]

    model = Model()
    model.setParam("OutputFlag", 0)

    Al = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Al")
    Bl = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Bl")
    Cl = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Cl")

    model.addConstrs(
        (Al * X[i] + Bl * Y[i] + Cl <= f(X[i], Y[i], tanh) for i in range(n)),
        name="ctr",
    )

    obj = LinExpr()
    obj = np.sum(f(X, Y, tanh)) - Al * np.sum(X) - Bl * np.sum(Y) - Cl * n
    if split_type == 11:
        obj -= f(lx, uy, tanh) - Al * lx - Bl * uy - Cl
    elif split_type == 12:
        obj -= f(ux, ly, tanh) - Al * ux - Bl * ly - Cl
    elif split_type == 21:
        obj -= f(ux, uy, tanh) - Al * ux - Bl * uy - Cl
    elif split_type == 22:
        obj -= f(lx, ly, tanh) - Al * lx - Bl * ly - Cl
    model.setObjective(obj, GRB.MINIMIZE)

    model.optimize()

    if model.status == GRB.Status.OPTIMAL:
        Al, Bl, Cl = model.getAttr("x", model.getVars())
        delta = get_LB_delta(Al, Bl, Cl, lx, ux, ly, uy, tanh)
        Cl += delta
        return Al, Bl, Cl, model.objVal / (n - 1)
    else:
        return None, None, None, None


def UB_split(lx, ux, ly, uy, tanh=True, split_type=0, n_samples=200):
    # Get upper bound plane with triangular domain.
    X = np.random.uniform(lx, ux, n_samples)
    Y = np.random.uniform(ly, uy, n_samples)

    if split_type == 11:
        sel = (ux - lx) * (Y - ly) <= (uy - ly) * (X - lx)
    elif split_type == 12:
        sel = (ux - lx) * (Y - ly) >= (uy - ly) * (X - lx)
    elif split_type == 21:
        sel = (ux - lx) * (Y - ly) <= (ly - uy) * (X - ux)
    elif split_type == 22:
        sel = (ux - lx) * (Y - ly) >= (ly - uy) * (X - ux)
    X, Y = X[sel], Y[sel]
    bndX = np.array([lx, lx, ux, ux])
    bndY = np.array([ly, uy, ly, uy])
    X = np.concatenate([bndX, X])
    Y = np.concatenate([bndY, Y])
    n = X.shape[0]

    model = Model()
    model.setParam("OutputFlag", 0)

    Au = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Au")
    Bu = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Bu")
    Cu = model.addVar(lb=-GRB.INFINITY, ub=GRB.INFINITY, name="Cu")

    model.addConstrs(
        (Au * X[i] + Bu * Y[i] + Cu >= f(X[i], Y[i], tanh) for i in range(n)),
        name="ctr",
    )

    obj = LinExpr()
    obj = Au * np.sum(X) + Bu * np.sum(Y) + Cu * n - np.sum(f(X, Y, tanh))
    if split_type == 11:
        obj += f(lx, uy, tanh) - Au * lx - Bu * uy - Cu
    elif split_type == 12:
        obj += f(ux, ly, tanh) - Au * ux - Bu * ly - Cu
    elif split_type == 21:
        obj += f(ux, uy, tanh) - Au * ux - Bu * uy - Cu
    elif split_type == 22:
        obj += f(lx, ly, tanh) - Au * lx - Bu * ly - Cu
    model.setObjective(obj, GRB.MINIMIZE)

    model.optimize()

    if model.status == GRB.Status.OPTIMAL:
        Au, Bu, Cu = model.getAttr("x", model.getVars())
        delta = get_UB_delta(Au, Bu, Cu, lx, ux, ly, uy, tanh)
        Cu -= delta
        return Au, Bu, Cu, model.objVal / (n - 1)
    else:
        return None, None, None, None


if __name__ == "__main__":
    print(LB(-1, 2, -2, 3))
    print(UB(-1, 2, -2, 3))
