"""Resolve release path tokens without embedding workstation paths."""
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]

def resolve_path(value: str) -> str:
    """Resolve data/root tokens and require explicit external tool locations."""
    data_dir = Path(os.environ.get("MISFIT_DATA_DIR", ROOT / "data")).expanduser().resolve()
    value = value.replace("<REPO_ROOT>/data/", str(data_dir) + "/")
    value = value.replace("<REPO_ROOT>", str(ROOT))
    for token, variable in (("<DATA_SHARE>", "MISFIT_DATA_SHARE"), ("<TOOLS>", "MISFIT_TOOLS")):
        if token in value:
            location = os.environ.get(variable)
            if not location:
                raise RuntimeError(f"Set {variable} to the local directory required by this driver.")
            value = value.replace(token, str(Path(location).expanduser().resolve()))
    if "<NILEARN_DATA>" in value:
        cache = os.environ.get("MISFIT_NILEARN_DATA", os.environ.get("NILEARN_DATA", str(Path.home() / "nilearn_data")))
        value = value.replace("<NILEARN_DATA>", str(Path(cache).expanduser().resolve()))
    return value


def reporting_reference(key: str, *, required: bool = True):
    """Read an external aggregate; optional missing references return None.

    Required figure inputs fail explicitly. An explicitly configured missing
    file or malformed JSON always fails, including for optional comparisons.
    """
    import json
    configured = os.environ.get("MISFIT_REPORTING_REFERENCES")
    path = Path(configured).expanduser() if configured else ROOT / "misfit/results/reporting_references.json"
    if not path.is_file():
        if not required and not configured:
            return None
        raise FileNotFoundError(
            "Set MISFIT_REPORTING_REFERENCES to an external reporting-references JSON file; numerical results are not bundled."
        )
    values = json.loads(path.read_text())
    if not isinstance(values, dict):
        raise ValueError("Reporting references must be a JSON object.")
    if key not in values:
        if not required:
            return None
        raise KeyError(f"Reporting reference key is missing: {key}")
    return values[key]
