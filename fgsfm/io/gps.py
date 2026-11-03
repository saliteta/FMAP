"""GPS priors: EXIF GPS from a BlocksExchange AT export -> local ENU (meters); DJI gimbal orientation priors."""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

_A, _E2 = 6378137.0, 6.69437999014e-3          # WGS84


def geodetic_to_ecef(lat_deg, lon_deg, h):
    lat, lon = np.radians(lat_deg), np.radians(lon_deg)
    N = _A / np.sqrt(1 - _E2 * np.sin(lat) ** 2)
    return np.stack([(N + h) * np.cos(lat) * np.cos(lon),
                     (N + h) * np.cos(lat) * np.sin(lon),
                     (N * (1 - _E2) + h) * np.sin(lat)], -1)


def geodetic_to_enu(lat, lon, h, lat0=None, lon0=None, h0=None) -> np.ndarray:
    lat0 = np.mean(lat) if lat0 is None else lat0
    lon0 = np.mean(lon) if lon0 is None else lon0
    h0 = np.mean(h) if h0 is None else h0
    d = geodetic_to_ecef(lat, lon, h) - geodetic_to_ecef(lat0, lon0, h0)
    la, lo = np.radians(lat0), np.radians(lon0)
    R = np.array([[-np.sin(lo), np.cos(lo), 0],
                  [-np.sin(la) * np.cos(lo), -np.sin(la) * np.sin(lo), np.cos(la)],
                  [np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)]])
    return d @ R.T


def read_exif_gps(xml_path: str | Path, cache: str | Path | None = None) -> dict[str, np.ndarray]:
    """image basename -> (lat, lon, alt) from <Photo><ExifData><GPS>, or from gps.npz next to xml_path when there is
    no AT export (written by scripts/prepare_dji_scene.py)."""
    scene_gps = Path(xml_path).parent / "gps.npz"
    if not Path(xml_path).exists() and scene_gps.exists():
        cache = scene_gps
    if cache is not None and Path(cache).exists():
        d = np.load(cache)
        return dict(zip(d["names"].tolist(), d["lla"]))
    out = {}
    name = None
    for _, el in ET.iterparse(str(xml_path), events=("end",)):
        if el.tag == "ImagePath":
            name = Path(el.text.replace("\\", "/")).name
        elif el.tag == "GPS" and name is not None:
            out[name] = np.array([float(el.find(k).text) for k in ("Latitude", "Longitude", "Altitude")])
        elif el.tag == "Photo":
            name = None
            el.clear()
        elif el.tag == "TiePoint":
            el.clear()
    if cache is not None:
        Path(cache).parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, names=np.array(list(out)), lla=np.stack(list(out.values())))
    return out


def gps_enu_for(names: list[str], xml_path: str | Path, cache: str | Path | None = None):
    """(n, 3) ENU positions for the given image names (NaN where missing) and a validity mask."""
    lla = read_exif_gps(xml_path, cache)
    arr = np.full((len(names), 3), np.nan)
    for i, n in enumerate(names):
        if n in lla:
            arr[i] = lla[n]
    ok = ~np.isnan(arr[:, 0])
    enu = np.full_like(arr, np.nan)
    enu[ok] = geodetic_to_enu(arr[ok, 0], arr[ok, 1], arr[ok, 2])
    return enu, ok


def w2c_rotation_from_gimbal(yaw_deg, pitch_deg) -> np.ndarray:
    """(n, 3, 3) world-to-camera rotations in ENU from DJI gimbal angles: yaw clockwise from north, pitch < 0 looks
    down, roll taken as 0 (DJI stabilises it). Camera x right, y down, z forward."""
    y, p = np.radians(np.asarray(yaw_deg, float)), np.radians(np.asarray(pitch_deg, float))
    f = np.stack([np.sin(y) * np.cos(p), np.cos(y) * np.cos(p), np.sin(p)], -1)
    r = np.stack([np.cos(y), -np.sin(y), np.zeros_like(y)], -1)
    return np.stack([r, np.cross(f, r), f], -2)


def gimbal_w2c_for(names: list[str], xml_path: str | Path):
    """(n, 3, 3) gimbal world-to-camera rotations (ENU) from the scene's gps.npz 'yaw_pitch' (written by
    scripts/prepare_dji_scene.py), or None when the scene has no gimbal data or an image is missing."""
    f = Path(xml_path).parent / "gps.npz"
    if not f.exists():
        return None
    d = np.load(f)
    if "yaw_pitch" not in d.files:
        return None
    idx = {n: i for i, n in enumerate(d["names"].tolist())}
    if any(n not in idx for n in names):
        return None
    yp = d["yaw_pitch"][[idx[n] for n in names]]
    return w2c_rotation_from_gimbal(yp[:, 0], yp[:, 1])


def flights_for(names: list[str], xml_path: str | Path):
    """(n,) int flight label per image from the scene's gps.npz 'flight' (scripts/prepare_dji_scene.py), or None.
    Each flight is one GPS session; its altitude can be offset by metres from the others."""
    f = Path(xml_path).parent / "gps.npz"
    if not f.exists():
        return None
    d = np.load(f)
    if "flight" not in d.files:
        return None
    idx = {n: i for i, n in enumerate(d["names"].tolist())}
    if any(n not in idx for n in names):
        return None
    _, lab = np.unique(d["flight"][[idx[n] for n in names]], return_inverse=True)
    return lab
