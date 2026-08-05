"""CSV Preprocessor for incoming dataset batch requests."""

from __future__ import annotations

import io
from typing import Any, Dict, List, Tuple
import pandas as pd
import numpy as np


class CSVPreprocessor:
    """Preprocesses user uploaded CSV data for 40-model batch predictions."""

    @staticmethod
    def process_csv(csv_source: bytes | str | io.BytesIO | pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Reads CSV input, extracts location & identity fields, aligns feature columns,
        and returns a list of processed row dictionary objects.
        """
        if isinstance(csv_source, pd.DataFrame):
            df = csv_source.copy()
        elif isinstance(csv_source, bytes):
            df = pd.read_csv(io.BytesIO(csv_source))
        elif isinstance(csv_source, (str, io.BytesIO)):
            df = pd.read_csv(csv_source)
        else:
            raise ValueError("Unsupported csv_source type")

        # Standardize index column
        if "system:index" not in df.columns:
            if "sample_id" in df.columns:
                df["system:index"] = df["sample_id"].astype(str)
            elif "index" in df.columns:
                df["system:index"] = df["index"].astype(str)
            else:
                df["system:index"] = [str(i) for i in range(len(df))]
        else:
            df["system:index"] = df["system:index"].astype(str)

        # Standardize lat / lon
        lat_col = next((c for c in ["latitude", "lat", "y"] if c in df.columns), None)
        lon_col = next((c for c in ["longitude", "lon", "x"] if c in df.columns), None)

        if lat_col:
            df["latitude"] = pd.to_numeric(df[lat_col], errors="coerce").fillna(0.0)
        else:
            df["latitude"] = 0.0

        if lon_col:
            df["longitude"] = pd.to_numeric(df[lon_col], errors="coerce").fillna(0.0)
        else:
            df["longitude"] = 0.0

        # Region fallback
        if "region" not in df.columns:
            regions = ["ayeyawaddy", "bago", "magway", "mandalay", "sagaing", "yangon"]
            region_series = pd.Series("", index=df.index)
            for r in regions:
                if f"region_{r}" in df.columns:
                    region_series = np.where(df[f"region_{r}"] == 1, r, region_series)
            df["region"] = np.where(region_series != "", region_series, "unknown")

        rows = []
        for i in range(len(df)):
            row_series = df.iloc[i]
            row_dict = row_series.to_dict()

            meta = {
                "index": str(row_dict.get("system:index", i)),
                "sample_id": str(row_dict.get("sample_id", row_dict.get("system:index", i))),
                "lat": float(row_dict.get("latitude", 0.0)),
                "lon": float(row_dict.get("longitude", 0.0)),
                "region": str(row_dict.get("region", "unknown")),
            }

            rows.append({
                "meta": meta,
                "features": row_dict
            })

        return rows


csv_preprocessor = CSVPreprocessor()
