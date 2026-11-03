"""LM step with the 3D points eliminated (Schur complement), matrix-free, for fgsfm/ba/full_intrinsics.py.

bae's LM forms H = J^T J over cameras, intrinsics AND every point (cuSPARSE SpGEMM), so its memory grows with the
number of points; bae's own Schur optimizer takes exactly two parameter groups (cameras, points) and cannot hold our
shared intrinsics. Here every parameter but the last (the points) forms the camera block c, the points form p:

    [U  W ] [dc]   [gc]          S dc = gc - W V^-1 gp,    S = U - W V^-1 W^T
    [W' V ] [dp] = [gp]          dp  = V^-1 (gp - W' dc)

V is block diagonal (one 3x3 per point). S is never formed: PCG on S only needs S x = Jc^T (Jc x - Jp V^-1 Jp^T Jc x)
plus the damping, and is preconditioned by the exact diagonal blocks of S (6x6 per pose, 3x3 intrinsics). Memory is
the Jacobian (linear in observations) plus a few vectors. Damping, clamping and step rejection follow bae's LM.
"""
from __future__ import annotations

import torch
from bae.autograd.function import TrackingTensor
from bae.autograd.graph import jacobian
from bae.optim import LM


def _blocks(J):
    """BSR Jacobian -> (row block, column block, values (nnzb, rows, cols)) per stored block."""
    crow = J.crow_indices()
    rb = torch.repeat_interleave(torch.arange(len(crow) - 1, device=crow.device), crow.diff())
    return rb, J.col_indices().long(), J.values()


def _bmv(A, x):
    return (A @ x.unsqueeze(-1)).squeeze(-1)


class _JacobianOp:
    """J @ D for the trust-region quality, with D = [dc, dp] and the Jacobian kept split."""

    def __init__(self, solver):
        self.s = solver

    def __matmul__(self, D):
        D = D.view(-1)
        nc = self.s.Jc.shape[1]
        return ((self.s.Jc @ D[:nc, None]).view(-1) + self.s.jp_apply(D[nc:].view(-1, 3)))[:, None]


class SchurLM(LM):
    def __init__(self, *args, cg_tol: float = 1e-5, cg_maxiter: int = 1000, **kwargs):
        super().__init__(*args, **kwargs)
        self.cg_tol, self.cg_maxiter = cg_tol, cg_maxiter
        self.cg_iters = []

    # --- point block: Jp has one 2x3 block per reprojection row block ---
    def jp_apply(self, w):                                   # (P, 3) -> residual-sized vector
        u = torch.zeros(self.nrb, 2, dtype=w.dtype, device=w.device)
        u[self.rb_p] = _bmv(self.Jp, w[self.pt])
        return u.view(-1)

    def jpt_apply(self, y):                                  # residual-sized vector -> (P, 3)
        out = torch.zeros(self.P, 3, dtype=y.dtype, device=y.device)
        return out.index_add_(0, self.pt, _bmv(self.Jp.mT, y.view(-1, 2)[self.rb_p]))

    def _pcg(self, matvec, b, minv):
        x = torch.zeros_like(b)
        r = b.clone()
        z = minv(r)
        p = z.clone()
        rz = r @ z
        bn = b.norm()
        for k in range(self.cg_maxiter):
            Ap = matvec(p)
            a = rz / (p @ Ap)
            x += a * p
            r -= a * Ap
            if r.norm() <= self.cg_tol * bn:
                break
            z = minv(r)
            rz, rz_old = r @ z, rz
            p = z + (rz / rz_old) * p
        self.cg_iters.append(k + 1)
        return x

    @torch.no_grad()
    def step(self, input, target=None, weight=None):
        for pg in self.param_groups:
            R = list(self.model(input))[0]
            J = jacobian(R, pg['params'])
            if isinstance(R, TrackingTensor):
                R = R.tensor()
            R = R.reshape(-1)
            self.nrb = R.numel() // 2
            *Jcs, Jp = J
            self.Jc = torch.cat([j.to_sparse_coo() for j in Jcs], dim=-1).to_sparse_csr()
            JcT = self.Jc.mT.to_sparse_csr()
            self.rb_p, self.pt, self.Jp = _blocks(Jp)
            self.P = Jp.shape[1] // 3
            dev, dt = R.device, R.dtype

            # undamped diagonals and the blocks needed for the preconditioner
            V = torch.zeros(self.P, 3, 3, dtype=dt, device=dev).index_add_(0, self.pt, self.Jp.mT @ self.Jp)
            pt_of_rb = torch.full((self.nrb,), -1, dtype=torch.long, device=dev)
            pt_of_rb[self.rb_p] = self.pt
            groups = []
            for j in Jcs:
                rb, col, vals = _blocks(j)
                n, b = j.shape[1] // vals.shape[-1], vals.shape[-1]
                U = torch.zeros(n, b, b, dtype=dt, device=dev).index_add_(0, col, vals.mT @ vals)
                has = pt_of_rb[rb] >= 0                                 # rows with a point (not GPS priors)
                pts = pt_of_rb[rb[has]]
                key, inv = torch.unique(col[has] * self.P + pts, return_inverse=True)
                W = torch.zeros(len(key), b, 3, dtype=dt, device=dev).index_add_(
                    0, inv, vals[has].mT @ self.Jp[torch.searchsorted(self.rb_p, rb[has])])
                groups.append((n, b, U, W, key // self.P, key % self.P))
            d_c = torch.cat([torch.diagonal(g[2], dim1=-2, dim2=-1).reshape(-1) for g in groups])
            d_p = torch.diagonal(V, dim1=-2, dim2=-1).reshape(-1)
            cd_c = d_c.clamp(pg['min'], pg['max'])
            cd_p = d_p.clamp(pg['min'], pg['max'])
            g_c = -(JcT @ R[:, None]).view(-1)
            g_p = -self.jpt_apply(R)

            self.last = self.loss = self.loss if hasattr(self, 'loss') else self.model.loss(input, target)
            self.reject_count = 0
            scale = 1.0
            while self.last <= self.loss:
                scale *= 1 + pg['damping']                              # LM: diag(A) *= 1 + damping, cumulative
                e_c = scale * cd_c - d_c
                e_p = (scale * cd_p - d_p).view(-1, 3)
                Vd = V + torch.diag_embed(e_p)
                Vi = torch.linalg.inv(Vd)

                def matvec(x):
                    y = (self.Jc @ x[:, None]).view(-1)
                    y = y - self.jp_apply(_bmv(Vi, self.jpt_apply(y)))
                    return (JcT @ y[:, None]).view(-1) + e_c * x

                # exact diagonal blocks of the damped S
                Pinv, off = [], 0
                for n, b, U, W, wc, wp in groups:
                    S = U + torch.diag_embed(e_c[off:off + n * b].view(n, b))
                    S = S.index_add(0, wc, -(W @ Vi[wp] @ W.mT))
                    Pinv.append(torch.linalg.inv(S))
                    off += n * b

                def minv(r):
                    out, o = [], 0
                    for Pi in Pinv:
                        n, b = Pi.shape[0], Pi.shape[1]
                        out.append(_bmv(Pi, r[o:o + n * b].view(n, b)).reshape(-1))
                        o += n * b
                    return torch.cat(out)

                rhs_c = g_c - (JcT @ self.jp_apply(_bmv(Vi, g_p))[:, None]).view(-1)
                D_c = self._pcg(matvec, rhs_c, minv)
                D_p = _bmv(Vi, g_p - self.jpt_apply((self.Jc @ D_c[:, None]).view(-1)))
                D = torch.cat([D_c, D_p.reshape(-1)])

                self.update_parameter(pg['params'], D)
                self.loss = self.model.loss(input, target)
                self.strategy.update(pg, last=self.last, loss=self.loss, J=_JacobianOp(self), D=D, R=R.view(-1, 1))
                if self.last < self.loss and self.reject_count < self.reject:  # reject step
                    self.update_parameter(params=pg['params'], step=-D)
                    self.loss, self.reject_count = self.last, self.reject_count + 1
                else:
                    break
            del self.Jc, self.Jp, JcT
        return self.loss
