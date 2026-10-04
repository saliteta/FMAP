"""Sequential sweep + seeded refinement for building a reliable scene graph (no GPS).

1. sweep:  windows of `window` images in name order, stride `stride`. Each window
           gives overlap measurements and keypoints; it is chained to the previous
           one through the shared cameras (robust Sim(3)). A failed window or an
           inconsistent link starts a new segment frame.
2. origin: the densest camera (sum of keypoint-predicted overlap within its
           segment) defines the world frame: origin camera = identity, its median
           depth = 1.
3. grow:   in rounds (until everything is joined or progress stalls), keypoint
           projection predicts overlap in the seed frame:
             - missing pairs (predicted, never co-batched)       -> densify batches
             - inconsistent pairs (measured vs predicted disagree) -> verify batches
             - cameras outside the seed (other segments / none)  -> link batches,
               anchors from VGGT's DINO descriptors and registered sequence neighbours
             - span batches: 8 cameras along the strongest correlation chain to a
               camera >= 4 hops away (multi-hop, long-baseline constraints)
           Each batch is registered from its seed anchors (existing poses are kept).
           A camera joins only if the batch measures it overlapping an agreeing anchor;
           a segment merges only when two batches agree on its Sim(3). Each round
           ends with coarse BA (poses + keypoints), which fuses all batches.
           When a round adds no camera, the number of overlapping cameras required
           steps down (3 -> 2 -> 1); what cannot link even with 1 is abandoned and
           left as an isolated segment / unregistered camera.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from fgsfm.fm.backbones.vggt_adapter import VGGTAdapter
from fgsfm.fm.coarse_ba import coarse_ba
from fgsfm.fm.keypoints import select_keypoints
from fgsfm.fm.overlap import batch_geometry, symmetric_overlap
from fgsfm.geometry.transforms import Sim3, centers, fit_sim3_from_cameras, rotation_angle_deg, to4x4
from fgsfm.graph.paths import max_product_paths, path_to
from fgsfm.graph.scene_graph import BatchOutput, SceneGraph


@dataclass
class BuilderConfig:
    window: int = 8
    stride: int = 4
    max_keypoints: int = 1000
    min_health: float = 5.0          # batch median depth confidence (failure detector)
    min_cam_conf: float = 3.0        # per-image median confidence to register that image
    rot_inlier_deg: float = 5.0
    max_rot_median_deg: float = 3.0
    max_scale_spread: float = 0.15   # MAD of per-pixel log depth ratios
    pred_thresh: float = 0.15        # predicted overlap that counts as an expected edge
    max_rounds: int = 20
    max_batches_per_round: int = 120
    span_batches: int = 20           # multi-hop batches per round
    span_min_hops: int = 4
    span_max_hops: int = 14
    span_min_weight: float = 0.2     # span paths only follow chains with correlation product >= this
    link_min_overlap: float = 0.1    # batch-measured overlap needed between a joining camera and seed anchors
                                     # (merge needs relax_merge_min_pairs such cross pairs)
    merge_agree_rot_deg: float = 2.0  # two batches must agree on the segment's Sim(3) ...
    merge_agree_spacing: float = 1.0  # ... and on its camera centers (in median camera spacings)
    # relaxation when a round adds no camera: overlapping-camera requirements step down to 1;
    # what still cannot link at the last level is abandoned
    relax_merge_min_cams: tuple = (3, 2, 1)
    relax_merge_min_pairs: tuple = (2, 1, 1)
    relax_new_cam_min_inliers: tuple = (3, 2, 1)
    link_anchors: int = 6
    max_link_attempts: int = 8
    ba_iters: int = 1500


@dataclass
class Log:
    events: list = field(default_factory=list)

    def __call__(self, **kw):
        kw["t"] = round(time.time(), 2)
        self.events.append(kw)


class GraphBuilder:
    def __init__(self, image_paths: list[Path], cfg: BuilderConfig = BuilderConfig(),
                 model: VGGTAdapter | None = None):
        self.paths = image_paths
        self.cfg = cfg
        self.model = model or VGGTAdapter()
        self.graph: SceneGraph | None = None
        self.log = Log()
        self.link_attempts = np.zeros(len(image_paths), int)
        self.relax = 0
        self.abandoned = np.zeros(len(image_paths), bool)
        self.rng = np.random.default_rng(0)
        self.history: list[dict] = []
        self.next_round = 0
        self.finished = False
        self.inference_time = 0.0
        self.gate_rejects = 0
        self.pending_merges: dict[int, list] = {}
        self.missing_pairs = 0

    # ------------------------------------------------------------ batches

    def run_batch(self, idx: np.ndarray, kind: str) -> BatchOutput:
        res = self.model.infer([self.paths[i] for i in idx], keep_images=True)
        self.inference_time += res.inference_time
        geom = batch_geometry(res.extrinsic, res.intrinsic, res.depth, res.depth_conf)
        kp = select_keypoints(geom, res.images, max_points=self.cfg.max_keypoints)
        S = len(idx)
        cam_conf = res.depth_conf.reshape(S, -1).median(1).values.cpu().numpy()
        b = BatchOutput(idx=np.asarray(idx), kind=kind, w2c=to4x4(res.extrinsic.cpu().numpy().astype(np.float64)),
                        K=res.intrinsic.cpu().numpy().astype(np.float64),
                        depth=res.depth[:, geom.ys, geom.xs].cpu().numpy().astype(np.float64),
                        valid=geom.valid.cpu().numpy(), overlap=symmetric_overlap(geom.covis).cpu().numpy(),
                        cam_conf=cam_conf, health=float(res.depth_conf.median()),
                        descriptors=res.descriptors.cpu().numpy(), keypoints=kp)
        if self.graph is None:
            self.graph = SceneGraph([p.name for p in self.paths], res.image_hw)
        self.graph.set_descriptors(b)
        return b

    def healthy(self, b: BatchOutput) -> bool:
        return b.health >= self.cfg.min_health

    def p(self, name: str):
        """Requirement at the current relaxation level."""
        sched = getattr(self.cfg, f"relax_{name}")
        return sched[min(self.relax, len(sched) - 1)]

    @property
    def max_relax(self) -> int:
        return len(self.cfg.relax_merge_min_cams) - 1

    def _fit(self, w2c_dst, w2c_src, log_ratios, min_inliers: int = 2):
        fit = fit_sim3_from_cameras(w2c_dst, w2c_src, log_ratios, rot_thresh_deg=self.cfg.rot_inlier_deg)
        n = len(w2c_dst)
        ok = (fit.num_inliers >= max(min_inliers, int(np.ceil(0.6 * n)))
              and np.median(fit.rot_res_deg[fit.inliers]) <= self.cfg.max_rot_median_deg
              and (np.isnan(fit.scale_spread) or fit.scale_spread <= self.cfg.max_scale_spread))
        return fit, ok

    def _connected(self, b: BatchOutput, seeds: np.ndarray, cands: np.ndarray) -> np.ndarray:
        """Candidates (local idx) reachable from seeds through batch-measured overlap >= link_min_overlap."""
        A = b.overlap >= self.cfg.link_min_overlap
        reach = np.zeros(len(b.idx), bool)
        reach[seeds] = True
        frontier = list(seeds)
        allowed = np.zeros(len(b.idx), bool)
        allowed[cands] = True
        allowed[seeds] = True
        while frontier:
            u = frontier.pop()
            for v in np.flatnonzero(A[u] & allowed & ~reach):
                reach[v] = True
                frontier.append(v)
        return np.array([c for c in cands if reach[c]], int)

    def _largest_component(self, b: BatchOutput, cands: np.ndarray) -> np.ndarray:
        best = np.zeros(0, int)
        left = list(cands)
        while left:
            comp = self._connected(b, np.array([left[0]]), np.array(left))
            comp = np.union1d(comp, [left[0]]).astype(int)
            if len(comp) > len(best):
                best = comp
            left = [c for c in left if c not in comp]
        return best

    # ------------------------------------------------------------ 1. sweep

    def sweep(self) -> None:
        n, w, s = len(self.paths), self.cfg.window, self.cfg.stride
        starts = list(range(0, max(n - w, 0) + 1, s))
        if starts[-1] + w < n:
            starts.append(n - w)
        cur = None
        for bid, st in enumerate(starts):
            idx = np.arange(st, min(st + w, n))
            b = self.run_batch(idx, "sweep")
            g = self.graph
            g.batches.append(dict(idx=idx.tolist(), kind="sweep", health=b.health))
            if not self.healthy(b):
                self.log(stage="sweep", batch=bid, start=st, status="unhealthy", health=b.health)
                continue
            g.add_measurement(b)
            ok_cams = np.flatnonzero(b.cam_conf >= self.cfg.min_cam_conf)
            status = "new_segment"
            if cur is not None:
                anc = np.array([l for l in ok_cams if g.frame_of[idx[l]] == cur])
                if len(anc) >= 2:
                    lr = g.log_depth_ratios(idx[anc], b.depth[anc], b.valid[anc])
                    fit, ok = self._fit(g.w2c[idx[anc]], b.w2c[anc], lr)
                    if ok:
                        # a window straddling a strip turn: cameras of the new view direction do not
                        # overlap the anchors, so they are left for the next segment / linking
                        new = np.array([l for l in ok_cams if g.frame_of[idx[l]] == -1], int)
                        new = self._connected(b, anc[fit.inliers], new)
                        g.register_batch(b, bid, cur, fit.sim3, new)
                        status = "chained"
            if status == "new_segment":
                new = np.array([l for l in ok_cams if g.frame_of[idx[l]] == -1], int)
                new = self._largest_component(b, new)
                if len(new) >= 2:
                    cur = g.new_frame()
                    g.register_batch(b, bid, cur, Sim3.identity(), new)
                else:
                    status = "skipped"
            self.log(stage="sweep", batch=bid, start=st, status=status, health=b.health, frame=cur)

    # ------------------------------------------------------------ 2. origin

    def choose_origin(self) -> int:
        g = self.graph
        best, best_d = -1, -1.0
        for f, cams in g.frames().items():
            if len(cams) < 3:
                continue
            cams_f, O = g.predicted_overlap(f)
            dens = O.sum(1) - 1.0
            k = int(np.argmax(dens))
            if dens[k] > best_d:
                best, best_d = int(cams_f[k]), float(dens[k])
        g.origin, g.seed_frame = best, int(g.frame_of[best])
        # world frame := origin camera frame, scaled so its median depth is 1
        Ro, to = g.w2c[best, :3, :3], g.w2c[best, :3, 3]
        d_med = float(np.median(g.depth[best][g.depth_valid[best]]))
        s = 1.0 / d_med
        g.transform_frame(g.seed_frame, Sim3(s, Ro, s * to))
        self.log(stage="origin", origin=best, name=g.names[best], density=best_d,
                 seed_frame=g.seed_frame, seed_size=len(g.cams_in(g.seed_frame)))
        return best

    # ------------------------------------------------------------ 3. grow

    def _integrate(self, b: BatchOutput, bid: int) -> str:
        """Register batch b into the seed frame (existing poses kept), merge segments, add keypoints."""
        g, seed, cfg = self.graph, self.graph.seed_frame, self.cfg
        idx = b.idx
        ok = b.cam_conf >= cfg.min_cam_conf
        anc = np.flatnonzero(ok & (g.frame_of[idx] == seed))
        if len(anc) < 2:
            return "no_anchors"
        lr = g.log_depth_ratios(idx[anc], b.depth[anc], b.valid[anc])
        fit, good = self._fit(g.w2c[idx[anc]], b.w2c[anc], lr)
        if not good:
            return "rejected"
        S = fit.sim3
        result = "updated"
        anc_in = anc[fit.inliers]                       # seed anchors that agree with the fit
        # merge another segment only if >= merge_min_cams of its cameras agree with this batch
        for G in np.unique(g.frame_of[idx[ok]]):
            if G < 0 or G == seed:
                continue
            gl = np.flatnonzero(ok & (g.frame_of[idx] == G))
            if len(gl) < self.p("merge_min_cams"):
                continue
            # the batch must actually see the segment and the seed overlapping, not just place them
            if (b.overlap[np.ix_(gl, anc_in)] >= cfg.link_min_overlap).sum() < self.p("merge_min_pairs"):
                self.gate_rejects += 1
                continue
            lr2 = -g.log_depth_ratios(idx[gl], S.s * b.depth[gl], b.valid[gl])
            k_min = self.p("merge_min_cams")
            fit2, good2 = self._fit(S.apply_w2c(b.w2c[gl]), g.w2c[idx[gl]], lr2, min_inliers=min(2, k_min))
            if good2 and fit2.num_inliers >= k_min:
                if self._merge_confirmed(int(G), fit2.sim3, bid):
                    size = len(g.cams_in(G))
                    g.transform_frame(int(G), fit2.sim3, new_frame=seed)
                    self.pending_merges.pop(int(G), None)
                    self.log(stage="grow", event="merge", batch=bid, frame=int(G), size=size)
                    result = "merged"
                else:
                    result = "merge_pending"
        # cameras with no pose at all: only from a well-anchored batch
        new = np.flatnonzero(ok & (g.frame_of[idx] == -1))
        new = self._connected(b, anc_in, new) if len(new) else new
        if len(new) and fit.num_inliers >= self.p("new_cam_min_inliers"):
            g.register_batch(b, bid, seed, S, new, add_keypoints=False)
            result = "registered" if result == "updated" else result
        g.register_batch(b, bid, seed, S, np.zeros(0, int), add_keypoints=True)
        return result

    def _merge_confirmed(self, G: int, sim: Sim3, bid: int) -> bool:
        """Merge a segment only when two different batches give agreeing Sim(3)s (seed <- G)."""
        g = self.graph
        CG = centers(g.w2c[g.cams_in(G)])
        Cs = centers(g.w2c[g.cams_in(g.seed_frame)])
        spacing = np.median(np.sort(np.linalg.norm(Cs[:, None] - Cs[None], axis=-1), axis=1)[:, 1])
        for other, obid in self.pending_merges.get(G, []):
            if obid == bid:
                continue
            d_rot = rotation_angle_deg(sim.R @ other.R.T)
            d_c = np.median(np.linalg.norm(sim.apply_points(CG) - other.apply_points(CG), axis=1))
            if d_rot <= self.cfg.merge_agree_rot_deg and d_c <= self.cfg.merge_agree_spacing * spacing:
                return True
        self.pending_merges.setdefault(G, []).append((sim, bid))
        return False

    def _link_anchors(self, qg: np.ndarray, cams: np.ndarray, O: np.ndarray, n_needed: int) -> list[int]:
        """Anchors for cameras outside the seed, interleaving two sources:
        descriptor similarity (same view direction) and registered sequence neighbours
        (adjacent captures; across strip turns these may look elsewhere, hence the mix).
        Retries rotate down the descriptor ranking."""
        g = self.graph
        seed_set = set(cams.tolist())
        seq = []
        for d in range(1, 4):
            for q in qg:
                for c in (q - d, q + d):
                    if c in seed_set and c not in seq:
                        seq.append(int(c))
        sim = (g.desc[qg] @ g.desc[cams].T).max(0)
        skip = int(self.link_attempts[qg].max()) * self.cfg.link_anchors // 2
        desc = [int(cams[k]) for k in np.argsort(-sim)[skip:skip + 3 * n_needed]]
        anchors: list[int] = []
        for a, b_ in zip(desc, seq + [None] * len(desc)):
            for c in (a, b_):
                if c is not None and c not in anchors:
                    anchors.append(c)
        return anchors[:n_needed]

    def plan_round(self, rng: np.random.Generator) -> list[tuple[str, np.ndarray]]:
        g, cfg, seed = self.graph, self.cfg, self.graph.seed_frame
        cams, O = g.predicted_overlap(seed)
        cob = g.cobatched[np.ix_(cams, cams)]
        M = np.nan_to_num(g.measured[np.ix_(cams, cams)])
        W = cfg.window
        plans: list[tuple[str, np.ndarray]] = []

        # links: everything outside the seed frame
        outside = np.flatnonzero((g.frame_of != seed) & ~self.abandoned & (self.link_attempts < cfg.max_link_attempts))
        groups_by_frame: dict[int, list[int]] = {}
        for c in outside:
            groups_by_frame.setdefault(int(g.frame_of[c]), []).append(int(c))
        seed_set = set(cams.tolist())
        for f, members in sorted(groups_by_frame.items(), key=lambda kv: -len(kv[1])):
            members = np.array(sorted(members))
            if f >= 0:
                # 3 consecutive members; prefer the end of the segment next to seed cameras
                near = [m for m in members if any((m + d) in seed_set or (m - d) in seed_set for d in (1, 2, 3))]
                if near:
                    q = near[int(self.link_attempts[near].argmin())]
                else:
                    q = members[int(np.argmax((g.desc[members] @ g.desc[cams].T).max(1)))]
                k = int(np.searchsorted(members, q))
                gsize = self.p("merge_min_cams")
                lo = max(0, min(k - 1, len(members) - gsize))
                groups = [members[lo:lo + gsize]]
                # second, independent triple for merge consensus: best descriptor match elsewhere
                rest = np.setdiff1d(members, groups[0])
                if len(rest) >= gsize:
                    q2 = rest[int(np.argmax((g.desc[rest] @ g.desc[cams].T).max(1)))]
                    k2 = int(np.searchsorted(rest, q2))
                    lo2 = max(0, min(k2 - 1, len(rest) - gsize))
                    groups.append(rest[lo2:lo2 + gsize])
            else:
                groups = [members[i:i + 2] for i in range(0, len(members), 2)]
            for qg in groups:
                anchors = self._link_anchors(qg, cams, O, W - len(qg))
                plans.append(("link", np.concatenate([qg, anchors]).astype(int)))
                self.link_attempts[qg] += 1

        # verify: measured and predicted overlap disagree
        inc = cob & (((M >= 0.2) & (O < 0.05)) | ((M < 0.05) & (O >= 0.3)))
        np.fill_diagonal(inc, False)
        for a in np.argsort(-inc.sum(1))[: max(1, cfg.max_batches_per_round // 6)]:
            if inc[a].sum() < 2:
                break
            nbr = [k for k in np.argsort(-O[a]) if k != a][: W - 1]
            plans.append(("verify", cams[[a] + nbr]))

        # span (multi-hop): cameras along the strongest correlation chain from a pivot to a far camera
        corr = np.where(cob, M, 0.0)
        for p in rng.choice(len(cams), min(cfg.span_batches, len(cams)), replace=False):
            w, _, parent, hops = max_product_paths(corr, np.array([p]), min_weight=cfg.span_min_weight)
            far = np.flatnonzero((hops >= cfg.span_min_hops) & (hops <= cfg.span_max_hops))
            if len(far) == 0:
                continue
            t = far[np.argmax(hops[far] - 1e-3 * w[far])]
            path = path_to(parent, int(t))
            sel = np.unique(np.round(np.linspace(0, len(path) - 1, min(W, len(path)))).astype(int))
            plans.append(("span", cams[np.array(path)[sel]]))

        # densify: predicted overlap but never co-batched
        miss = (~cob) & (O >= cfg.pred_thresh)
        np.fill_diagonal(miss, False)
        covered = np.zeros_like(miss)
        ii, jj = np.nonzero(np.triu(miss))
        for k in np.argsort(-O[ii, jj]):
            if len(plans) >= cfg.max_batches_per_round:
                break
            a, c = ii[k], jj[k]
            if covered[a, c]:
                continue
            members = [a, c]
            while len(members) < W:
                gain = (miss[:, members] & ~covered[:, members]).sum(1) + O[:, members].min(1)
                gain[members] = -1
                nxt = int(np.argmax(gain))
                if gain[nxt] <= cfg.pred_thresh:
                    break
                members.append(nxt)
            covered[np.ix_(members, members)] = True
            plans.append(("densify", cams[members]))
        self.missing_pairs = int(np.triu(miss).sum())
        return plans[: cfg.max_batches_per_round]

    def grow(self, eval_fn=None) -> list[dict]:
        """Growth rounds (round index, rng and history live on the builder)."""
        g, cfg = self.graph, self.cfg
        rng, history = self.rng, self.history
        for r in range(self.next_round, cfg.max_rounds):
            size0, frames0 = int((g.frame_of == g.seed_frame).sum()), len(g.frames())
            plans = self.plan_round(rng)
            counts: dict[str, int] = {}
            self.gate_rejects = 0
            for kind, idx in plans:
                b = self.run_batch(np.asarray(idx), kind)
                bid = len(g.batches)
                g.batches.append(dict(idx=list(map(int, idx)), kind=kind, health=b.health))
                if not self.healthy(b):
                    counts[f"{kind}:unhealthy"] = counts.get(f"{kind}:unhealthy", 0) + 1
                    continue
                g.add_measurement(b)
                res = self._integrate(b, bid)
                counts[f"{kind}:{res}"] = counts.get(f"{kind}:{res}", 0) + 1
            ba = self.bundle_adjust()
            size1, frames1 = int((g.frame_of == g.seed_frame).sum()), len(g.frames())
            rec = dict(round=r + 1, relax_level=self.relax, planned=len(plans), outcomes=counts,
                       seed_size=size1, segments=frames1, unregistered=int((g.frame_of < 0).sum()),
                       missing_pairs=self.missing_pairs, overlap_gate_rejects=self.gate_rejects, ba=ba)
            if eval_fn is not None:
                rec["eval"] = eval_fn(self, f"round{r + 1}")
            history.append(rec)
            self.log(stage="grow", **{k: v for k, v in rec.items() if k != "eval"})
            self.next_round = r + 1
            done = False
            if size1 == g.n:
                done = self.missing_pairs == 0 or not plans
            elif size1 == size0 and frames1 == frames0:          # no camera added this round
                if self.relax < self.max_relax:
                    self.relax += 1
                    self.link_attempts[:] = 0                  # retry everything outside with the new level
                    self.log(stage="grow", event="relax", level=self.relax,
                             merge_min_cams=self.p("merge_min_cams"))
                    print(f"  -> no camera added: relax level {self.relax} "
                          f"(overlapping cameras needed: {self.p('merge_min_cams')})", flush=True)
                else:
                    self.abandoned[g.frame_of != g.seed_frame] = True
                    self.log(stage="grow", event="abandon", cameras=int(self.abandoned.sum()))
                    print(f"  -> still nothing at 1 overlapping camera: abandoning {int(self.abandoned.sum())} cameras",
                          flush=True)
                    done = True
            self.finished = done
            if done:
                break
        return history

    # ------------------------------------------------------------ BA

    def bundle_adjust(self) -> dict:
        g, seed = self.graph, self.graph.seed_frame
        cams = g.cams_in(seed)
        X, _, _, obs_kp, obs_cam, obs_uv, _ = g.kp.gather(seed)
        loc = np.full(g.n, -1)
        loc[cams] = np.arange(len(cams))
        m = loc[obs_cam] >= 0
        obs_kp, obs_cam, obs_uv = obs_kp[m], loc[obs_cam[m]], obs_uv[m]
        nobs = np.bincount(obs_kp, minlength=len(X))
        m = nobs[obs_kp] >= 2
        if m.sum() < 100:
            return dict(skipped=True)
        res = coarse_ba(g.w2c[cams], g.K[cams], X, obs_cam[m], obs_kp[m], obs_uv[m],
                        fixed_cam=int(loc[g.origin]), iters=self.cfg.ba_iters)
        g.w2c[cams] = res.w2c
        g.kp.scatter(seed, res.X)
        return dict(reproj_before_px=round(res.reproj_before_px, 3), reproj_after_px=round(res.reproj_after_px, 3),
                    num_obs=res.num_obs, num_points=res.num_points, num_cams=len(cams))
