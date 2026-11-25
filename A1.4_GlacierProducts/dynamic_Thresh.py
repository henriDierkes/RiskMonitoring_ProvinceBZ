"""Glacier outline pipeline: S2 + S1 coherence -> masks -> cleaned masks -> polygons -> areas.
Coherence thresholds are read per year from a text file (same for ascending and descending).
"""
import re
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import skimage as ski
from rasterio.features import shapes
from rasterio.io import MemoryFile
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.warp import Resampling, reproject
from shapely.geometry import box, shape

# rasterio/NumPy 2.5 warning ignore
warnings.filterwarnings(
    "ignore", message="Setting the shape on a NumPy array", category=DeprecationWarning
)

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------
YEARS = [2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025, 2026]
VERSION = "v0.3_full"
MIN_PATCH_M2 = 10_000        # remove isolated patches smaller than this
MAX_HOLE_M2 = 1   # fill holes smaller than this
TAG = f"minPatch{MIN_PATCH_M2}_hole{MAX_HOLE_M2}"   

STEPS = {"composite", "threshold", "align", "clean", "polygons", "areas"}

BASE = Path("/mnt/CEPH_PROJECTS/provinzBZ_risk_EO/GlacierOutlines")
OUT = BASE / "results_dynamicThresh"
S2_BASE = BASE / "sentinel_data/S2_indices_test"
S1_BASE = BASE / "sentinel_data/S1_coherence"
INVENTORY = BASE / "references/glacier_inventory/SGIhom_1997_v2025.shp"
THR_FILE = BASE / "sentinel_data/S1_coherence/coherence_thresholds_full.txt"

# layer -> folder + filename templates
LAYERS = {
    "s2": dict(
        folder="S2",
        aligned="s2_aligned_{y}_{t}",
        clean="s2_glacier_{y}_{t}_{tag}",
        outline="s2_glacier_outlines_{y}_{v}_{tag}",
    ),
    "s1": dict(
        folder="S1",
        aligned="s1_aligned_{y}_{t}",
        clean="s1_glacier_{y}_{t}_{tag}",
        outline="s1_glacier_outlines_{y}_{v}_{tag}",
    ),
    "combined": dict(
        folder="S1_S2",
        aligned="s1_s2_asc_desc_combined_{y}_{t}",
        clean="s1_s2_asc_desc_combined_glacier_{y}_{t}_{tag}",
        outline="s1_s2_glacier_outlines_{y}_{v}_{tag}",
    ),
}


# ----------------------------------------------------------------------------
# Per-year thresholds
# ----------------------------------------------------------------------------
def load_thresholds(path=THR_FILE):
    """Parse lines like '2016 threshold = 0.40' -> {2016: 0.40}.
    The same threshold is used for ascending and descending."""
    table = {}
    for line in path.read_text().splitlines():
        m = re.match(r"\s*(\d{4})\s*threshold\s*=\s*([\d.]+)", line)
        if m:
            table[int(m[1])] = float(m[2])
    return table


THRESHOLDS = load_thresholds()


def get_thr(year):
    """Coherence threshold for a year, or None if missing from the file."""
    return THRESHOLDS.get(year)


def thr_tag(year):
    """Short tag for filenames, e.g. '0.40'."""
    return f"{get_thr(year):.2f}"


def layer_path(layer, kind, year, ext):
    cfg = LAYERS[layer]
    name = cfg[kind].format(y=year, t=thr_tag(year), v=VERSION, tag=TAG)
    return OUT / cfg["folder"] / str(year) / f"{name}.{ext}"


# ----------------------------------------------------------------------------
# I/O helpers
# ----------------------------------------------------------------------------
def find_one(folder, pattern):
    hits = sorted(folder.glob(pattern))
    return hits[0] if len(hits) == 1 else None


def read_band(path):
    with rasterio.open(path) as src:
        return src.read(1), src.meta.copy()


def write_tif(path, arr, crs, transform, nodata=0):
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
        count=1, dtype="uint8", crs=crs, transform=transform,
        nodata=nodata, compress="lzw",
    ) as dst:
        dst.write(arr.astype("uint8"), 1)
    print(f"Saved: {path}")


# ----------------------------------------------------------------------------
# 1) S2 composite from east + west mosaics
# ----------------------------------------------------------------------------
def merged_max(east, west):
    with rasterio.open(east) as e, rasterio.open(west) as w:
        arr, tf = merge([e, w], method="max")
        return arr[0], tf, e.crs


def step_composite(year):
    d = S2_BASE / str(year)
    o = S2_BASE / str(year)
    f = {
        (reg, idx): find_one(d, f"S2_mosaic_{reg}_*_{idx}.tiff")
        for reg in ("east", "west") for idx in ("ndsi_0.6", "ratio_10")
    }
    if None in f.values():
        print(f"Skipping composite {year}: expected exactly two input files.")
        return
    ndsi, tf, crs = merged_max(f["east", "ndsi_0.6"], f["west", "ndsi_0.6"])
    #ratio, tf, crs = merged_max(f["east", "ratio_1"], f["west", "ratio_1"])
    glacier = (ndsi == 1).astype("uint8")
    write_tif(o / f"glacier_composite_{year}_{VERSION}.tiff", glacier, crs, tf, nodata=255)


# ----------------------------------------------------------------------------
# 2) Threshold S1 coherence (ascending / descending)
# ----------------------------------------------------------------------------
def threshold(src_path, dst_path, thr):
    arr, meta = read_band(src_path)
    data = arr.astype("float32")
    invalid = np.isnan(data)
    if meta["nodata"] is not None and not np.isnan(meta["nodata"]):
        invalid |= data == meta["nodata"]
    out = (data < thr).astype("uint8")      # 1 = low coherence (glacier)
    out[invalid] = 255
    write_tif(dst_path, out, meta["crs"], meta["transform"], nodata=255)


def step_threshold(year):
    folder = S1_BASE / str(year)
    for orbit in ("ascending", "descending"):
        src = find_one(folder, f"coh_mosaic_{orbit}_{year}.tiff")
        if src is None:
            print(f"Missing {orbit} coherence for {year}, skipping.")
            continue
        thr = get_thr(year)
        print(f"{year} {orbit}: threshold = {thr:.2f}")
        threshold(src, folder / f"{orbit}_{year}_thresh_{thr:.2f}.tiff", thr)


# ----------------------------------------------------------------------------
# 3) Align everything to the S1 grid and combine
# ----------------------------------------------------------------------------
def align(arr, meta, ref, src_nodata=None, dst_nodata=0):
    dst = np.full((ref["height"], ref["width"]), dst_nodata, dtype=arr.dtype)
    reproject(
        source=arr, destination=dst,
        src_transform=meta["transform"], src_crs=meta["crs"], src_nodata=src_nodata,
        dst_transform=ref["transform"], dst_crs=ref["crs"], dst_nodata=dst_nodata,
        resampling=Resampling.nearest,
    )
    return dst


def step_align(year):
    s2_file = S2_BASE / str(year) / f"glacier_composite_{year}_{VERSION}.tiff"
    s1_files = [S1_BASE / str(year) / f"{o}_{year}_thresh_{get_thr(year):.2f}.tiff"
                for o in ("ascending", "descending")]
    missing = [p.name for p in (s2_file, *s1_files) if not p.exists()]
    if missing:
        print(f"Missing for {year}: {missing}, skipping.")
        return

    (asc, asc_meta), (desc, desc_meta) = (read_band(p) for p in s1_files)
    s2, s2_meta = read_band(s2_file)
    ref = asc_meta

    s2_bin = align(s2, s2_meta, ref, dst_nodata=0) == 1
    s1_bin = (
        (align(asc, asc_meta, ref, 255, 255) == 1)
        | (align(desc, desc_meta, ref, 255, 255) == 1)
    )

    for layer, arr in {"s2": s2_bin, "s1": s1_bin, "combined": s2_bin | s1_bin}.items():
        write_tif(layer_path(layer, "aligned", year, "tiff"),
                  arr, ref["crs"], ref["transform"], nodata=None)


# ----------------------------------------------------------------------------
# 4) Clip to 1997 outlines, remove small patches, fill small holes
# ----------------------------------------------------------------------------
def clean_mask(binary, transform, crs, outline):
    """`outline` must already be in `crs`. Returns (cleaned uint8, cropped transform)."""
    meta = dict(driver="GTiff", height=binary.shape[0], width=binary.shape[1],
                count=1, dtype="uint8", crs=crs, transform=transform)
    with MemoryFile() as mem:
        with mem.open(**meta) as tmp:
            tmp.write(binary.astype("uint8"), 1)
        with mem.open() as src:
            clipped, tf = mask(src, outline.geometry, crop=True, filled=False)

    inside = ~np.ma.getmaskarray(clipped[0])         # True inside glacier polygons
    binary = clipped[0].filled(0).astype(bool)

    # m² -> pixels (assumes a projected CRS)
    px_area = abs(tf.a * tf.e)
    min_px = max(1, round(MIN_PATCH_M2 / px_area))
    hole_px = max(1, round(MAX_HOLE_M2 / px_area))

    # max_size removes objects <= value, so subtract 1 to keep "smaller than" semantics
    cleaned = ski.morphology.remove_small_objects(binary, max_size=min_px - 1, connectivity=2)
    cleaned = ski.morphology.remove_small_holes(cleaned, max_size=hole_px - 1, connectivity=2)
    return (cleaned & inside).astype("uint8"), tf


def step_clean(year, inventory):
    outline = None
    for layer in LAYERS:
        src = layer_path(layer, "aligned", year, "tiff")
        if not src.exists():
            print(f"Missing {src.name}, skipping.")
            continue
        arr, meta = read_band(src)
        if outline is None:                          # reproject once per year
            outline = inventory.to_crs(meta["crs"])
        cleaned, tf = clean_mask(arr, meta["transform"], meta["crs"], outline)
        write_tif(layer_path(layer, "clean", year, "tiff"), cleaned, meta["crs"], tf)


# ----------------------------------------------------------------------------
# 5) Raster -> polygons
# ----------------------------------------------------------------------------
def step_polygons(year):
    for layer in LAYERS:
        src = layer_path(layer, "clean", year, "tiff")
        if not src.exists():
            print(f"Missing {src.name}, skipping.")
            continue
        arr, meta = read_band(src)
        geoms = [shape(g) for g, _ in shapes(arr, mask=arr == 1, transform=meta["transform"])]
        gdf = gpd.GeoDataFrame({"value": [1] * len(geoms)}, geometry=geoms, crs=meta["crs"])
        out = layer_path(layer, "outline", year, "shp")
        out.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_file(out)
        print(f"{layer}: {len(gdf)} polygons -> {out}")


# ----------------------------------------------------------------------------
# 6) Area per layer and region (east / west S2 extent)
# ----------------------------------------------------------------------------
# Fixed study-area extents (EPSG:32632)
SPATIAL_EXTENTS = {
    "west": {"west": 610950, "east": 674650, "south": 5143820, "north": 5206380, "crs": "EPSG:32632"},  # Western South Tyrol glaciers
    "east": {"west": 704100, "east": 747040, "south": 5195880, "north": 5219660, "crs": "EPSG:32632"},  # Eastern South Tyrol glaciers
}


def get_region_extents():
    """Build the east/west clip boxes from the fixed coordinates."""
    return {
        region: gpd.GeoDataFrame(
            geometry=[box(e["west"], e["south"], e["east"], e["north"])],
            crs=e["crs"],
        )
        for region, e in SPATIAL_EXTENTS.items()
    }


def step_areas(year, extents):
    rows = []
    for layer in LAYERS:
        path = layer_path(layer, "outline", year, "shp")
        if not path.exists():
            print(f"Missing {path.name}, skipping.")
            continue
        gdf = gpd.read_file(path)
        if gdf.crs is None or not gdf.crs.is_projected:
            raise ValueError(f"{path.name}: CRS is not projected, area would be wrong.")
        for region, ext in extents.items():
            clipped = gpd.clip(gdf, ext.to_crs(gdf.crs))
            area = clipped.geometry.area.sum() / 1e6
            rows.append(dict(year=year, layer=layer, region=region,
                             n_polygons=len(clipped), area_km2=round(area, 3)))
            print(f"{year} {layer:<9} {region:<5} {area:8.2f} km² ({len(clipped)} polygons)")
    return rows

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    inventory = gpd.read_file(INVENTORY) if "clean" in STEPS else None
    extents = get_region_extents() if "areas" in STEPS else None
    rows = []
    for year in YEARS:
        print(f"\n===== {year} =====")
        if "composite" in STEPS:
            step_composite(year)

        if get_thr(year) is None:
            print(f"No coherence threshold for {year} in {THR_FILE.name}, "
                  f"skipping remaining steps.")
            continue
        print(f"Threshold {year}: {thr_tag(year)}")

        if "threshold" in STEPS:
            step_threshold(year)
        if "align" in STEPS:
            step_align(year)
        if "clean" in STEPS:
            step_clean(year, inventory)
        if "polygons" in STEPS:
            step_polygons(year)
        if "areas" in STEPS:
            rows += step_areas(year, extents)

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(OUT / f"glacier_area_long_{VERSION}_{TAG}.csv", index=False)
        wide = df.pivot_table(index=["layer", "region"], columns="year",
                              values="area_km2").reset_index()
        wide.to_csv(OUT / f"glacier_area_wide_{VERSION}_{TAG}.csv", index=False)
        print("\n", wide.to_string(index=False))


if __name__ == "__main__":
    main()