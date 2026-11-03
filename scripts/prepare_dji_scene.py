"""Raw DJI drone photos -> a scene folder the pipeline reads (GauUscene layout, without a ground-truth model).

    python scripts/prepare_dji_scene.py --src <folder with the JPGs, any depth> --out <scene>/colmap_metrics

Writes
  images/<name>   undistorted with the lens calibration in each photo's XMP (drone-dji:DewarpData), same size,
                  PINHOLE with the calibrated focal length and the principal point at the image centre
  gps.npz         (names, lla, yaw_pitch): XMP latitude, longitude, ellipsoidal altitude and gimbal yaw / pitch
                  (degrees), flight (source sub-folder) -> read by fgsfm/io/gps.py (GPS positions, gimbal rotation
                  priors and per-flight GPS offsets for the cluster merge)
  sparse/0/       cameras.txt + images.txt as a pseudo-reference from GPS + gimbal yaw / pitch. The scripts need a
                  reference model for the image list and their (evaluation-only) metrics; against this one the
                  metrics are meter / degree level and only catch gross failures.

Assumes one camera (one DewarpData for all photos) and unique file names across subfolders.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fgsfm.io.gps import geodetic_to_enu, w2c_rotation_from_gimbal  # noqa: E402

XMP = ["GpsLatitude", "GpsLongitude", "AbsoluteAltitude", "GimbalYawDegree", "GimbalPitchDegree", "DewarpData"]
_maps = None


def read_xmp(p: Path) -> dict:
    b = open(p, "rb").read(300000).decode("latin1")
    d = {"path": str(p), "name": p.name}
    for k in XMP:
        m = re.search(rf'drone-dji:{k}="([^"]*)"', b)
        d[k] = m.group(1) if m else None
    return d


def calibration(dewarp: str, W: int, H: int):
    """DJI DewarpData 'date;fx,fy,cx,cy,k1,k2,p1,p2,k3' (cx, cy are offsets from the image centre)."""
    fx, fy, cx, cy, k1, k2, p1, p2, k3 = [float(v) for v in dewarp.split(";")[1].split(",")]
    K = np.array([[fx, 0, W / 2 + cx], [0, fy, H / 2 + cy], [0, 0, 1]])
    K_new = np.array([[fx, 0, W / 2], [0, fx, H / 2], [0, 0, 1]])
    return K, np.array([k1, k2, p1, p2, k3]), K_new


def undistort(job):
    global _maps
    src, dst, K, dist, K_new, W, H = job
    if Path(dst).exists():
        return
    if _maps is None:
        _maps = cv2.initUndistortRectifyMap(K, dist, None, K_new, (W, H), cv2.CV_16SC2)
    im = cv2.imread(src)
    assert im.shape[:2] == (H, W), (src, im.shape)
    cv2.imwrite(dst, cv2.remap(im, *_maps, cv2.INTER_LINEAR), [cv2.IMWRITE_JPEG_QUALITY, 95])


def w2c_from_gimbal(c, yaw, pitch):
    R = w2c_rotation_from_gimbal(yaw, pitch)
    return R, -R @ c


def quat(R):
    w = np.sqrt(max(0.0, 1 + np.trace(R))) / 2
    x = np.copysign(np.sqrt(max(0.0, 1 + R[0, 0] - R[1, 1] - R[2, 2])) / 2, R[2, 1] - R[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] + R[1, 1] - R[2, 2])) / 2, R[0, 2] - R[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1 - R[0, 0] - R[1, 1] + R[2, 2])) / 2, R[1, 0] - R[0, 1])
    return np.array([w, x, y, z])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="<scene>/colmap_metrics")
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    files = sorted(p for p in args.src.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg"))
    with ThreadPoolExecutor(16) as ex:
        meta = sorted(ex.map(read_xmp, files), key=lambda m: m["name"])
    names = [m["name"] for m in meta]
    assert len(set(names)) == len(names), "image names must be unique across subfolders"
    missing = [m["name"] for m in meta if None in (m["GpsLatitude"], m["GimbalYawDegree"], m["DewarpData"])]
    assert not missing, f"{len(missing)} photos without DJI GPS / gimbal / DewarpData XMP: {missing[:3]}"
    dewarps = {m["DewarpData"] for m in meta}
    assert len(dewarps) == 1, f"{len(dewarps)} different lens calibrations; expects one camera"
    H, W = cv2.imread(meta[0]["path"]).shape[:2]
    K, dist, K_new = calibration(dewarps.pop(), W, H)
    print(f"{len(meta)} photos {W}x{H}, f={K[0, 0]:.2f}, distortion {dist.round(5).tolist()}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    lla = np.array([[float(m["GpsLatitude"]), float(m["GpsLongitude"]), float(m["AbsoluteAltitude"])] for m in meta])
    yaw_pitch = np.array([[float(m["GimbalYawDegree"]), float(m["GimbalPitchDegree"])] for m in meta])
    flight = np.array([str(Path(m["path"]).parent.relative_to(args.src)) for m in meta])   # one GPS session each
    np.savez(args.out / "gps.npz", names=np.array(names), lla=lla, yaw_pitch=yaw_pitch, flight=flight)
    sp = args.out / "sparse" / "0"
    sp.mkdir(parents=True, exist_ok=True)
    enu = geodetic_to_enu(lla[:, 0], lla[:, 1], lla[:, 2])
    (sp / "cameras.txt").write_text(f"1 PINHOLE {W} {H} {K_new[0, 0]} {K_new[1, 1]} {K_new[0, 2]} {K_new[1, 2]}\n")
    lines = []
    for i, (m, c) in enumerate(zip(meta, enu)):
        R, t = w2c_from_gimbal(c, float(m["GimbalYawDegree"]), float(m["GimbalPitchDegree"]))
        lines += [f"{i + 1} {' '.join(f'{v:.9f}' for v in quat(R))} {' '.join(f'{v:.6f}' for v in t)} 1 {m['name']}", ""]
    (sp / "images.txt").write_text("\n".join(lines) + "\n")
    (sp / "points3D.txt").write_text("")
    (args.out / "prepare.json").write_text(json.dumps(dict(src=str(args.src), images=len(meta), width=W, height=H,
                                                           K_dist=K.tolist(), dist=dist.tolist(), K=K_new.tolist())))
    print(f"pseudo-reference + gps.npz written; ENU extent {np.ptp(enu, 0).round(1)} m", flush=True)

    (args.out / "images").mkdir(exist_ok=True)
    jobs = [(m["path"], str(args.out / "images" / m["name"]), K, dist, K_new, W, H) for m in meta]
    with ProcessPoolExecutor(args.workers) as ex:
        for k, _ in enumerate(ex.map(undistort, jobs, chunksize=8)):
            if k % 500 == 0:
                print(f"undistorted {k}/{len(jobs)}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
