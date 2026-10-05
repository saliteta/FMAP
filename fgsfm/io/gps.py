"""GPS priors: EXIF GPS from a BlocksExchange AT export -> local ENU (meters)."""
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
    """image basename -> (lat, lon, alt) from <Photo><ExifData><GPS>."""
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
