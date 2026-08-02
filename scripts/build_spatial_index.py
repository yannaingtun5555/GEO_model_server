#!/usr/bin/env python3
"""Build the row-aligned serving spatial index from QA pipeline Parquet outputs.

The trained feature matrix intentionally excludes identifiers and coordinates.  Serving
must therefore keep a separate, immutable locator table with exactly the same row order.
This command validates that alignment against every shared, non-null feature before it
writes the index.  It refuses to produce an index when any region or row is misaligned.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


REGION_FILES = {
    "ayeyawaddy": "gee_2018_2026/ayeyawaddy_agri_suitability_with_infra.parquet",
    "bago": "gee_bago_2018_2026/bago_agri_suitability_with_infra.parquet",
    "magway": "gee_magway_2018_2026/magway_agri_suitability_with_infra.parquet",
    "mandalay": "gee_mandalay_2018_2026/mandalay_agri_suitability_with_infra.parquet",
    "sagaing": "gee_sagaing_2018_2026/sagaing_agri_suitability_with_infra.parquet",
    "yangon": "gee_yangon_2018_2026/yangon_agri_suitability_with_infra.parquet",
}

LOCATOR_COLUMNS = [
    "sample_id",
    "grid_id",
    "year_month",
    "longitude",
    "latitude",
    "data_source",
    "source_date",
    "source_version",
    "quality_flag",
]

ALIGNMENT_COLUMNS = [
    "elevation_m",
    "slope_degrees",
    "aspect_degrees",
    "distance_to_surface_water_m",
    "soil_cec_cmol_kg_0_30cm",
    "soil_clay_pct_0_30cm",
    "soil_sand_pct_0_30cm",
    "soil_silt_pct_0_30cm",
    "soil_soc_g_kg_0_30cm",
    "soil_ph_h2o_0_30cm",
    "surface_water_occurrence_pct",
    "surface_water_seasonality_months",
    "distance_to_road_km",
    "road_density_km_per_sqkm",
    "distance_to_railway_km",
    "railway_density_km_per_sqkm",
    "distance_to_river_km",
    "river_density_km_per_sqkm",
    "urban_fraction",
    "builtup_fraction",
    "cropland_fraction",
    "non_cropland_fraction",
    "permanent_water_fraction",
    "population_density",
    "valid_agriculture_mask",
    "chirps_precipitation_mm",
    "mean_temperature_c",
    "solar_radiation_mj_m2_day",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_alignment(
    features: pd.DataFrame,
    source: pd.DataFrame,
    region: str,
    offset: int,
) -> None:
    feature_slice = features.iloc[offset : offset + len(source)].reset_index(drop=True)
    if len(feature_slice) != len(source):
        raise RuntimeError(
            f"{region}: feature matrix ended early at offset {offset}; "
            f"expected {len(source):,} rows, found {len(feature_slice):,}"
        )

    expected_month = source["year_month"].astype(str).str[-2:].astype(int).to_numpy()
    actual_month = feature_slice["data_month"].astype(int).to_numpy()
    if not np.array_equal(expected_month, actual_month):
        raise RuntimeError(f"{region}: data_month does not align with year_month")

    one_hot = f"region_{region}"
    if one_hot not in feature_slice or not (feature_slice[one_hot].to_numpy() == 1).all():
        raise RuntimeError(f"{region}: region one-hot boundary is not aligned")

    for column in ALIGNMENT_COLUMNS:
        if column not in source or column not in feature_slice:
            continue
        source_values = pd.to_numeric(source[column], errors="coerce").to_numpy(float)
        feature_values = pd.to_numeric(feature_slice[column], errors="coerce").to_numpy(float)
        observed = np.isfinite(source_values)
        if observed.any() and not np.isclose(
            feature_values[observed],
            source_values[observed],
            rtol=1e-5,
            atol=1e-6,
        ).all():
            mismatch_count = int(
                (~np.isclose(
                    feature_values[observed],
                    source_values[observed],
                    rtol=1e-5,
                    atol=1e-6,
                )).sum()
            )
            raise RuntimeError(
                f"{region}: {column} has {mismatch_count:,} row-alignment mismatches"
            )


def compact_features(features: pd.DataFrame) -> pd.DataFrame:
    """Write model-compatible float32/small-int features for bounded serving RSS."""
    compact = features.copy()
    for column in compact.columns:
        if pd.api.types.is_float_dtype(compact[column].dtype):
            compact[column] = pd.to_numeric(compact[column], downcast="float")
        elif pd.api.types.is_integer_dtype(compact[column].dtype):
            compact[column] = pd.to_numeric(compact[column], downcast="integer")
    return compact


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Path to myanmar-agri-geo-csv-pipeline/data/output",
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=project_root / "data" / "processed" / "features_dataset.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "data" / "processed" / "spatial_index.parquet",
    )
    parser.add_argument(
        "--serving-features-output",
        type=Path,
        default=project_root / "data" / "processed" / "features_serving.parquet",
        help="Compact, row-identical numeric matrix used only by inference",
    )
    args = parser.parse_args()

    source_root = args.source_root.expanduser().resolve()
    features_path = args.features.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    serving_features_path = args.serving_features_output.expanduser().resolve()
    if not features_path.is_file():
        raise SystemExit(f"Feature matrix not found: {features_path}")

    features = pd.read_parquet(features_path)
    locator_frames: list[pd.DataFrame] = []
    source_artifacts: list[dict[str, object]] = []
    offset = 0

    for region, relative_path in REGION_FILES.items():
        source_path = source_root / relative_path
        if not source_path.is_file():
            raise SystemExit(f"Required regional source is missing: {source_path}")
        source = pd.read_parquet(source_path)
        missing = [column for column in LOCATOR_COLUMNS if column not in source]
        if missing:
            raise RuntimeError(f"{region}: missing locator columns: {missing}")
        validate_alignment(features, source, region, offset)
        locator_frame = source[LOCATOR_COLUMNS].copy()
        locator_frame.insert(3, "region", region)
        locator_frames.append(locator_frame)
        source_artifacts.append(
            {
                "region": region,
                "filename": source_path.name,
                "rows": len(source),
                "sha256": sha256_file(source_path),
            }
        )
        offset += len(source)

    if offset != len(features):
        raise RuntimeError(
            f"Regional sources contain {offset:,} rows but feature matrix contains {len(features):,}"
        )

    spatial_index = pd.concat(locator_frames, ignore_index=True)
    if spatial_index["sample_id"].duplicated().any():
        raise RuntimeError("sample_id must be unique across the serving spatial index")
    if not spatial_index["latitude"].between(9, 29).all():
        raise RuntimeError("spatial index contains latitude outside Myanmar bounds")
    if not spatial_index["longitude"].between(92, 102).all():
        raise RuntimeError("spatial index contains longitude outside Myanmar bounds")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    spatial_index.to_parquet(temporary_path, index=False)
    os.replace(temporary_path, output_path)

    compact = compact_features(features)
    serving_features_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_features_path = serving_features_path.with_suffix(
        serving_features_path.suffix + ".tmp"
    )
    compact.to_parquet(temporary_features_path, index=False)
    os.replace(temporary_features_path, serving_features_path)

    metadata = {
        "schema_version": "spatial-index-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rows": len(spatial_index),
        "feature_rows": len(features),
        "source_feature_dataset_sha256": sha256_file(features_path),
        "feature_dataset_sha256": sha256_file(serving_features_path),
        "feature_dataset_filename": serving_features_path.name,
        "feature_numeric_storage": "float32_and_lossless_downcast_integers",
        "spatial_index_sha256": sha256_file(output_path),
        "alignment_validation": "all shared non-null values exact within rtol=1e-5",
        "source_artifacts": source_artifacts,
    }
    metadata_path = output_path.with_name("spatial_index_metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(
        f"Wrote {len(spatial_index):,} verified locator rows to {output_path} "
        f"({output_path.stat().st_size / 1024 / 1024:.1f} MiB) and compact features to "
        f"{serving_features_path} "
        f"({serving_features_path.stat().st_size / 1024 / 1024:.1f} MiB)"
    )


if __name__ == "__main__":
    main()
