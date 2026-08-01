import csv
import os
from pathlib import Path
from typing import List, Dict

def _read_csv(path: Path) -> List[Dict[str, str]]:
    """Read a CSV file into a list of dictionaries (string values)."""
    with path.open(newline='') as f:
        reader = csv.DictReader(f)
        return [row for row in reader]

def load_static(path: Path) -> List[Dict[str, str]]:
    """Load static dataset (CSV). Returns list of dict rows."""
    return _read_csv(path)

def load_dynamic(path: Path) -> List[Dict[str, str]]:
    """Load dynamic dataset (CSV). Returns list of dict rows."""
    return _read_csv(path)

def merge_datasets(static_rows: List[Dict[str, str]],
                   dynamic_rows: List[Dict[str, str]],
                   join_key: str) -> List[Dict[str, str]]:
    """Merge static and dynamic rows on *join_key* (inner join)."""
    # Build lookup for static rows
    static_lookup = {row[join_key]: row for row in static_rows}
    merged = []
    for dyn in dynamic_rows:
        key = dyn.get(join_key)
        static_part = static_lookup.get(key, {})
        combined = {**dyn, **static_part}
        merged.append(combined)
    return merged

def filter_columns(rows: List[Dict[str, str]], required_cols: List[str]) -> List[Dict[str, str]]:
    """Keep only *required_cols* (if present) in each row."""
    filtered = []
    for row in rows:
        filtered.append({col: row.get(col, '') for col in required_cols})
    return filtered

def generate_labels(rows: List[Dict[str, str]], label_rules: dict) -> List[Dict[str, str]]:
    """Apply simple rule‑based labeling to create prediction columns.
    The *label_rules* dict follows the same schema used in config.yaml.
    """
    for row in rows:
        for col, rule in label_rules.items():
            if rule["type"] == "threshold":
                src_val = float(row.get(rule["source"], 0) or 0)
                if rule.get("direction") == "above":
                    row[col] = "1" if src_val > rule["value"] else "0"
                else:
                    row[col] = "1" if src_val <= rule["value"] else "0"
            elif rule["type"] == "categorical":
                src_val = row.get(rule["source"], "")
                row[col] = rule["mapping"].get(src_val, rule.get("default", ""))
            else:
                # Unknown rule type – leave empty
                row[col] = ""
    return rows

def save_processed(rows: List[Dict[str, str]], out_path: Path):
    """Write processed rows to a CSV file (header based on first row)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with out_path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
