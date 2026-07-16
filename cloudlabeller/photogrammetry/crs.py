# CloudLabeller — photogrammetric reconstruction and bidirectional 2D <-> 3D
# point-cloud labelling with U-Net label propagation.
# Copyright (C) 2026 Ítalo Gomes Gonçalves
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General
# Public License for more details.
#
# You should have received a copy of the GNU General Public License along
# with this program. If not, see <https://www.gnu.org/licenses/>.
#
# Commercial licensing: this program is also available under a separate
# commercial license from the author — see README.md.

"""Coordinate reference systems: local ENU frame -> projected CRS (pyproj).

The app's internal frame stays local metric ENU (clouds are float32 and VTK
jitters at UTM magnitudes); conversion to a projected CRS happens only at
export time, in float64. COLMAP's ``model_aligner`` anchors the ENU frame at
the **first** reference GPS coordinate (the first line of ``ref_images.txt``
— verified in COLMAP's ``ConvertCameraLocations``), so the exact export
transform needs that origin, not the mean GPS.

Heights are ellipsoidal (WGS84) unless the EGM96 geoid conversion is
requested, which needs a small PROJ grid (~3 MB, downloaded once from
cdn.proj.org into the user's PROJ data directory).
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np

log = logging.getLogger("cloudlabeller.crs")

GEOID_EPSG = 5773                               # EGM96 height (grid ~3 MB)
ORIGIN_CONVENTION_FIRST_GPS = "first_gps_image"  # COLMAP model_aligner ENU anchor

# When several UTM variants cover the site, suggest the modern national datum
# first (SIRGAS 2000 ≈ WGS 84 in practice, but it is the legal standard in
# Latin America); WGS 84 always exists as the final fallback.
_PREFERRED_DATUMS = ("SIRGAS 2000", "ETRS89", "NAD83(2011)", "NAD83",
                     "GDA2020", "GDA94", "WGS 84")

TransformFn = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class EnuOrigin:
    lla: tuple[float, float, float]   # (lat, lon, alt), degrees / metres
    exact: bool                       # True = known first-GPS convention


def resolve_enu_origin(geo: dict, workspace: str | Path,
                       images_dir: str | Path | None = None) -> EnuOrigin:
    """The ENU origin of a georeferenced project.

    New projects store the exact origin (``origin_convention`` ==
    ``first_gps_image``). Legacy projects stored the *mean* camera GPS, which
    is NOT where COLMAP anchored the frame — recover the true origin by
    replaying georef_cli's ref-file order: the first ``cameras.json`` entry
    whose image has EXIF GPS. Falls back to the stored mean (marked inexact)
    if the images are gone.
    """
    if geo.get("origin_convention") == ORIGIN_CONVENTION_FIRST_GPS:
        return EnuOrigin(tuple(float(v) for v in geo["origin_lla"]), True)

    from cloudlabeller.photogrammetry.georef import exif_gps
    from cloudlabeller.photogrammetry.pipeline import (
        CAMERAS_FILE,
        _default_images_dir,
    )

    ws = Path(workspace)
    img_dir = Path(images_dir) if images_dir is not None else _default_images_dir(ws)
    cams_path = ws / CAMERAS_FILE
    if cams_path.exists():
        for cam in json.loads(cams_path.read_text(encoding="utf-8")):
            name = cam.get("name") or Path(cam["path"]).name
            lla = exif_gps(img_dir / name)
            if lla is not None:
                return EnuOrigin(lla, True)
    log.warning("could not recover the exact ENU origin (images missing?) — "
                "using the stored mean GPS; exports may be offset by up to "
                "the site extent")
    return EnuOrigin(tuple(float(v) for v in geo["origin_lla"]), False)


def suggest_projected_epsg(origin_lla) -> int:
    """EPSG code of the UTM zone covering the site, preferring the modern
    national datum (e.g. SIRGAS 2000 in Brazil); WGS 84 UTM as fallback."""
    from pyproj.aoi import AreaOfInterest
    from pyproj.database import query_utm_crs_info

    lat, lon = float(origin_lla[0]), float(origin_lla[1])
    infos = [i for i in query_utm_crs_info(
                 area_of_interest=AreaOfInterest(west_lon_degree=lon,
                                                 south_lat_degree=lat,
                                                 east_lon_degree=lon,
                                                 north_lat_degree=lat))
             if not i.deprecated]
    for datum in _PREFERRED_DATUMS:
        for info in infos:
            if info.name.startswith(datum + " /"):
                return int(info.code)
    if infos:
        return int(infos[0].code)
    zone = min(max(int((lon + 180.0) // 6) + 1, 1), 60)   # polar/no-DB fallback
    return (32600 if lat >= 0 else 32700) + zone


@lru_cache(maxsize=1)
def projected_crs_catalogue() -> tuple[tuple[int, str], ...]:
    """All non-deprecated EPSG projected CRS as (code, name), name-sorted.
    ~5300 entries, ~0.05 s — cheap enough to build once per session."""
    from pyproj.database import query_crs_info

    infos = query_crs_info(auth_name="EPSG", pj_types="PROJECTED_CRS")
    return tuple(sorted(((int(i.code), i.name) for i in infos if not i.deprecated),
                        key=lambda item: item[1].lower()))


def crs_display(epsg: int, name: str | None = None) -> str:
    if name is None:
        import pyproj
        name = pyproj.CRS.from_epsg(int(epsg)).name
    return f"EPSG:{int(epsg)} — {name}"


def _compound_crs(epsg: int):
    """Projected CRS + EGM96 orthometric height."""
    import pyproj

    return pyproj.CRS.from_user_input(f"EPSG:{int(epsg)}+{GEOID_EPSG}")


def _geoid_group(epsg: int):
    import pyproj

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")   # pyproj warns about missing grids
        return pyproj.transformer.TransformerGroup(
            pyproj.CRS.from_epsg(4979), _compound_crs(epsg), always_xy=True)


def geoid_ready(epsg: int) -> bool:
    """True if the PROJ grids for EGM96 orthometric heights are installed."""
    return bool(_geoid_group(epsg).best_available)


def refresh_grid_cache() -> None:
    """Make freshly downloaded PROJ grids visible to this thread.

    PROJ caches grid availability in the (thread-local) context, so a grid
    downloaded moments ago is still reported missing by new transformers on
    the same thread; re-setting the data dir forces a re-scan (verified
    2026-07-16: same-thread check stays False after download without this).
    """
    import pyproj.datadir

    pyproj.datadir.set_data_dir(pyproj.datadir.get_data_dir())


def download_geoid_grids(epsg: int) -> None:
    """Fetch the missing PROJ grid(s) (EGM96 ~3 MB) from cdn.proj.org into
    the user PROJ directory. One-time; needs network."""
    group = _geoid_group(epsg)
    if group.best_available:
        return
    group.download_grids(verbose=False)
    refresh_grid_cache()
    if not _geoid_group(epsg).best_available:
        raise RuntimeError(
            "The EGM96 geoid grid could not be downloaded (offline?) — "
            "export with ellipsoidal heights, or retry with a connection.")


def _transform_arrays(transformer, x, y, z, **kwargs):
    """pyproj sends 1-element arrays through a deprecated scalar path
    (NumPy DeprecationWarning, future error) — pad to 2 and slice back."""
    if len(x) == 1:
        out = transformer.transform(np.repeat(x, 2), np.repeat(y, 2),
                                    np.repeat(z, 2), **kwargs)
        return tuple(v[:1] for v in out)
    return transformer.transform(x, y, z, **kwargs)


def _enu_lla_pipeline(origin_lla):
    """Transformer whose forward direction maps local ENU -> geodetic
    (lon, lat, h — degrees/metres, WGS84) about the origin; INVERSE goes
    back. Matches COLMAP's EllipsoidToENU (WGS84 topocentric)."""
    import pyproj

    lat0, lon0, alt0 = (float(v) for v in origin_lla)
    to_ecef = pyproj.Transformer.from_crs("EPSG:4979", "EPSG:4978",
                                          always_xy=True)
    # 1-element arrays: pyproj's scalar path hits a NumPy deprecation.
    ex, ey, ez = to_ecef.transform([lon0], [lat0], [alt0])
    x0, y0, z0 = float(ex[0]), float(ey[0]), float(ez[0])
    return pyproj.Transformer.from_pipeline(
        f"+proj=pipeline "
        f"+step +inv +proj=topocentric +ellps=WGS84 "
        f"+X_0={x0!r} +Y_0={y0!r} +Z_0={z0!r} "
        f"+step +inv +proj=cart +ellps=WGS84")


def lla_to_enu_transform(origin_lla) -> TransformFn:
    """(N, 3) geodetic (lat, lon, alt) -> (N, 3) float64 local ENU about the
    origin — the georeferencing direction (export uses the inverse)."""
    from pyproj.enums import TransformDirection

    pipeline = _enu_lla_pipeline(origin_lla)

    def fn(lla: np.ndarray) -> np.ndarray:
        lla = np.asarray(lla, np.float64).reshape(-1, 3)
        east, north, up = _transform_arrays(
            pipeline, lla[:, 1], lla[:, 0], lla[:, 2],
            direction=TransformDirection.INVERSE)
        return np.column_stack([east, north, up])

    return fn


def enu_to_crs_transform(origin_lla, epsg: int, orthometric: bool = False
                         ) -> tuple[TransformFn, "object"]:
    """Build the export transform for a georeferenced project.

    Returns ``(fn, crs)``: ``fn`` maps an (N, 3) local-ENU array to (N, 3)
    float64 coordinates in the target CRS; ``crs`` is the ``pyproj.CRS``
    actually used (compound projected+EGM96 when ``orthometric``).

    Path: ENU -> ECEF (inverse topocentric about the origin, WGS84 — matching
    COLMAP's EllipsoidToENU) -> geodetic -> target CRS.
    """
    import pyproj

    enu_to_lla = _enu_lla_pipeline(origin_lla)

    if orthometric:
        group = _geoid_group(epsg)
        if not group.best_available:
            raise RuntimeError(
                "Orthometric heights need the EGM96 geoid grid — call "
                "download_geoid_grids() first (one-time, ~3 MB).")
        crs = _compound_crs(epsg)
        lla_to_crs = group.transformers[0]     # sorted by accuracy, available
    else:
        crs = pyproj.CRS.from_epsg(int(epsg))
        lla_to_crs = pyproj.Transformer.from_crs("EPSG:4979", crs, always_xy=True)

    def fn(xyz: np.ndarray) -> np.ndarray:
        xyz = np.asarray(xyz, np.float64)
        out = np.empty(xyz.shape, np.float64)
        chunk = 2_000_000
        for i in range(0, len(xyz), chunk):
            j = min(i + chunk, len(xyz))
            lon, lat, h = _transform_arrays(
                enu_to_lla, xyz[i:j, 0], xyz[i:j, 1], xyz[i:j, 2])
            east, north, up = _transform_arrays(lla_to_crs, lon, lat, h)
            out[i:j, 0], out[i:j, 1], out[i:j, 2] = east, north, up
        return out

    return fn, crs


def frame_labels(geo: dict | None) -> tuple[str, str]:
    """(short, full) description of the project's coordinate frame — the
    short form fits the status bar, the full form is its tooltip.

    ``geo`` is ``settings["georeferenced"]`` (or None); once reprojected it
    carries ``crs = {epsg, name, orthometric, offset}`` and stored
    coordinates are the projected ones minus ``offset``.
    """
    if not geo:
        return ("no CRS", "Local frame — arbitrary units, not georeferenced.")
    crs_info = geo.get("crs")
    if not crs_info:
        return ("local ENU",
                "Local East-North-Up frame — metres, true north, origin at "
                "the site (georeferenced; not projected). Use Georeferencing "
                "→ Reproject to CRS… for map coordinates.")
    short = f"EPSG:{crs_info['epsg']}"
    full = f"EPSG:{crs_info['epsg']} — {crs_info.get('name', '?')}"
    full += (" · EGM96 (sea-level) heights" if crs_info.get("orthometric")
             else " · ellipsoidal heights")
    off = crs_info.get("offset") or (0.0, 0.0, 0.0)
    if any(off):
        short += " (offset)"
        full += (f"\nStored/displayed coordinates are offset by "
                 f"({off[0]:.0f}, {off[1]:.0f}, {off[2]:.0f}) to preserve "
                 f"precision; exports write the full coordinates.")
    return (short, full)


def choose_offset(coords: np.ndarray) -> np.ndarray:
    """Km-rounded X/Y offset near the site so stored float32 coordinates
    stay small (Z stays absolute — elevations are already small)."""
    offset = np.floor(np.median(np.asarray(coords, np.float64), axis=0)
                      / 1000.0) * 1000.0
    offset[2] = 0.0
    return offset


def _checked_transformer(src, dst):
    """Best transformer src->dst, refusing the silent 'ballpark' fallback
    that pyproj uses when a datum/geoid grid is missing."""
    import pyproj

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        group = pyproj.transformer.TransformerGroup(src, dst, always_xy=True)
    if not group.best_available:
        raise RuntimeError(
            "A PROJ grid needed for this conversion is not installed — "
            "download the geoid grid first (or keep ellipsoidal heights).")
    return group.transformers[0]


def project_frame_transform(geo: dict, origin_lla, target_epsg: int,
                            orthometric: bool = False
                            ) -> tuple[TransformFn, "object"]:
    """Exact mapping from the project's *stored* coordinates to absolute
    coordinates in the target CRS.

    Source frame comes from ``geo``: local ENU (about ``origin_lla``) for a
    merely georeferenced project, or CRS-minus-offset once reprojected.
    Returns ``(fn, target_crs)`` like :func:`enu_to_crs_transform`.
    """
    crs_info = geo.get("crs")
    if not crs_info:
        return enu_to_crs_transform(origin_lla, target_epsg, orthometric)

    import pyproj

    src = (pyproj.CRS.from_user_input(f"EPSG:{crs_info['epsg']}+{GEOID_EPSG}")
           if crs_info.get("orthometric")
           else pyproj.CRS.from_epsg(int(crs_info["epsg"])))
    dst = (_compound_crs(target_epsg) if orthometric
           else pyproj.CRS.from_epsg(int(target_epsg)))
    transformer = _checked_transformer(src, dst)
    offset = np.asarray(crs_info.get("offset") or (0.0, 0.0, 0.0), np.float64)

    def fn(xyz: np.ndarray) -> np.ndarray:
        xyz = np.asarray(xyz, np.float64) + offset
        x, y, z = _transform_arrays(transformer, xyz[:, 0], xyz[:, 1],
                                    xyz[:, 2])
        return np.column_stack([x, y, z])

    return fn, dst


def crs_prj_wkt(crs) -> str:
    """WKT for a ``.prj`` sidecar — ESRI WKT1 (the .prj convention) when the
    CRS supports it, WKT2 otherwise (compound CRS often can't do WKT1)."""
    from pyproj.enums import WktVersion

    try:
        wkt = crs.to_wkt(WktVersion.WKT1_ESRI)
        if wkt:
            return wkt
    except Exception:
        pass
    return crs.to_wkt(WktVersion.WKT2_2019)
