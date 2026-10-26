"""Metadata and alignment audit; no training, outcomes, identifiers, or array exports.

Object-string NPY members are decoded with a non-executing pickle opcode parser.
No pickle callable, import, or NumPy object constructor is executed by that parser.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import pickletools
import re
import zipfile

import numpy as np

from misfit.paths import resolve_path

ROOT = Path(resolve_path("<REPO_ROOT>/data/oasis3"))
INPUT = ROOT / "oasis3_features.npz"
CNN = ROOT / "oasis3_cnn18_arrays.npz"
KIT = ROOT / "misfit_revision_kit_20261011.zip"
CDR_MEMBER = "3_evidence/provenance/oasis3_cdr_participant_arrays.npz"


def npy_header(stream):
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        return np.lib.format.read_array_header_1_0(stream)
    if version == (2, 0):
        return np.lib.format.read_array_header_2_0(stream)
    raise ValueError("Unsupported NPY header version")


def inert_object_strings(payload: bytes, expected_shape: tuple[int, ...]):
    """Interpret a bounded subset of pickle as inert data, never execute it."""
    if len(payload) > 2_000_000 or len(expected_shape) != 1:
        raise ValueError("Unexpected object-array size or dimensionality")
    stack, memo = [], {}
    marker = object()
    globals_seen = set()
    allowed_globals = {
        ("numpy.core.multiarray", "_reconstruct"),
        ("numpy._core.multiarray", "_reconstruct"),
        ("numpy", "ndarray"),
        ("numpy", "dtype"),
    }

    def marked():
        pos = next((i for i in range(len(stack) - 1, -1, -1)
                    if stack[i] is marker), None)
        if pos is None:
            raise ValueError("Missing pickle mark")
        values = stack[pos + 1:]
        del stack[pos:]
        return values

    for opcode, arg, _ in pickletools.genops(payload):
        name = opcode.name
        if name in {"PROTO", "FRAME"}:
            continue
        if name in {"SHORT_BINUNICODE", "BINUNICODE", "UNICODE", "SHORT_BINBYTES", "BINBYTES",
                    "BININT", "BININT1", "BININT2", "LONG1", "LONG4", "INT", "LONG"}:
            stack.append(arg)
        elif name == "NONE":
            stack.append(None)
        elif name in {"NEWTRUE", "NEWFALSE"}:
            stack.append(name == "NEWTRUE")
        elif name == "MARK":
            stack.append(marker)
        elif name == "EMPTY_TUPLE":
            stack.append(())
        elif name == "EMPTY_LIST":
            stack.append([])
        elif name == "TUPLE":
            stack.append(tuple(marked()))
        elif name in {"TUPLE1", "TUPLE2", "TUPLE3"}:
            count = int(name[-1])
            values = tuple(stack[-count:])
            del stack[-count:]
            stack.append(values)
        elif name == "APPENDS":
            values = marked()
            if not isinstance(stack[-1], list):
                raise ValueError("APPENDS target is not a list")
            stack[-1].extend(values)
        elif name == "APPEND":
            value = stack.pop()
            stack[-1].append(value)
        elif name == "MEMOIZE":
            memo[len(memo)] = stack[-1]
        elif name in {"BINPUT", "LONG_BINPUT", "PUT"}:
            memo[int(arg)] = stack[-1]
        elif name in {"BINGET", "LONG_BINGET", "GET"}:
            stack.append(memo[int(arg)])
        elif name in {"GLOBAL", "STACK_GLOBAL"}:
            if name == "GLOBAL":
                module, symbol = arg.split(" ", 1)
            else:
                symbol, module = stack.pop(), stack.pop()
            if (module, symbol) not in allowed_globals:
                raise ValueError("Unexpected pickle symbol")
            globals_seen.add((module, symbol))
            stack.append({"kind": "symbol", "module": module, "name": symbol})
        elif name == "REDUCE":
            args, symbol = stack.pop(), stack.pop()
            if not isinstance(symbol, dict) or symbol.get("kind") != "symbol":
                raise ValueError("Unexpected REDUCE target")
            stack.append({"kind": "inert_reduction", "symbol": symbol, "args": args})
        elif name == "BUILD":
            state = stack.pop()
            if not isinstance(stack[-1], dict) or stack[-1].get("kind") != "inert_reduction":
                raise ValueError("Unexpected BUILD target")
            stack[-1]["state"] = state
        elif name == "STOP":
            break
        else:
            raise ValueError("Unsupported pickle opcode: " + name)
        if len(stack) > 10_000 or len(memo) > 10_000:
            raise ValueError("Object metadata exceeds safety bounds")
    if len(stack) != 1 or not isinstance(stack[0], dict):
        raise ValueError("Invalid object array envelope")
    result = stack[0]
    if result.get("symbol", {}).get("name") != "_reconstruct":
        raise ValueError("Invalid NumPy array envelope")
    state = result.get("state")
    if not isinstance(state, tuple) or len(state) != 5 or state[1] != expected_shape:
        raise ValueError("Object-array shape mismatch")
    values = state[4]
    if not isinstance(values, list) or len(values) != expected_shape[0] or not all(type(x) is str for x in values):
        raise ValueError("Object array contains non-string values")
    return np.asarray(values, dtype=str), sorted(".".join(x) for x in globals_seen)


def load_inputs(input_path=INPUT, cnn_path=CNN, clinical_path=None, clinical_kit=KIT):
    """Return aligned arrays in memory only, for local authorized validation."""
    schema, strings, symbols = {}, {}, {}
    with zipfile.ZipFile(input_path) as z:
        for name in z.namelist():
            with z.open(name) as stream:
                shape, fortran, dtype = npy_header(stream)
                key = name[:-4]
                schema[key] = {"shape": list(shape), "dtype": str(dtype),
                               "object_dtype": bool(dtype.hasobject), "fortran_order": fortran}
                if dtype.hasobject:
                    strings[key], symbols[key] = inert_object_strings(stream.read(), shape)
    with np.load(input_path, allow_pickle=False) as data:
        numeric = {key: data[key].copy() for key in schema if not schema[key]["object_dtype"]}
    with np.load(cnn_path, allow_pickle=False) as data:
        cnn = {key: data[key].copy() for key in ["sessions", "label", "site", "scanners"]}
    if clinical_path is not None:
        cdr_bytes = Path(clinical_path).read_bytes()
    else:
        with zipfile.ZipFile(clinical_kit) as z:
            cdr_bytes = z.read(CDR_MEMBER)
    with np.load(io.BytesIO(cdr_bytes), allow_pickle=False) as data:
        cdr = {key: data[key].copy() for key in ["sessions", "label", "scanner", "cdr_raw", "age", "sex_male1"]}
    sessions, subjects = strings["sessions"], strings["subjects"]
    days = np.asarray([int(re.search(r"_d(\d+)$", s).group(1)) for s in sessions])
    keep = np.asarray(sorted(np.where(subjects == subject)[0][np.argmin(days[subjects == subject])]
                             for subject in np.unique(subjects)))
    lookup = {session: i for i, session in enumerate(sessions)}
    if len(lookup) != len(sessions):
        raise ValueError("Duplicate source sessions")
    aligned_rows = np.asarray([lookup[session] for session in cnn["sessions"]])
    if not np.array_equal(aligned_rows, keep):
        raise ValueError("Canonical cohort no longer matches earliest-session selection")
    checks = [
        np.array_equal(numeric["label"][aligned_rows], cnn["label"]),
        np.array_equal(strings["scanner"][aligned_rows], cnn["scanners"][cnn["site"]]),
        np.array_equal(cdr["sessions"], cnn["sessions"]),
        np.array_equal(cdr["label"], numeric["label"][aligned_rows]),
        np.array_equal(cdr["scanner"], strings["scanner"][aligned_rows]),
        np.array_equal(cdr["age"], numeric["age"][aligned_rows]),
        np.array_equal(cdr["label"], (cdr["cdr_raw"] > 0).astype(int)),
        len(np.unique(subjects[aligned_rows])) == len(aligned_rows),
        bool(np.isfinite(numeric["features"][aligned_rows]).all()),
    ]
    if not all(checks):
        raise ValueError("Input alignment, finite-feature, or clinical consistency check failed")
    return dict(schema=schema, strings=strings, symbols=symbols, numeric=numeric,
                cnn=cnn, cdr=cdr, cdr_bytes=cdr_bytes, earliest_rows=keep,
                aligned_rows=aligned_rows, X=numeric["features"][aligned_rows],
                y=numeric["label"][aligned_rows], site=cnn["site"], cdr_raw=cdr["cdr_raw"])


def audit():
    data = load_inputs()
    rows, keep = data["aligned_rows"], data["earliest_rows"]
    cnn, cdr, strings, numeric = (data[k] for k in ["cnn", "cdr", "strings", "numeric"])
    def equal(a, b):
        return bool(np.array_equal(a, b))
    unique_subjects, counts = np.unique(strings["subjects"], return_counts=True)
    return {
        "audit_date": "2026-10-18",
        "input_path": str(INPUT), "bytes": INPUT.stat().st_size,
        "sha256": hashlib.sha256(INPUT.read_bytes()).hexdigest(),
        "schema": data["schema"],
        "safe_loading": {"numeric_allow_pickle": False,
                         "object_arrays": "Non-executing pickletools opcode parser; symbols represented as inert dictionaries, no REDUCE/BUILD call executed.",
                         "allowed_symbols_seen": data["symbols"],
                         "identifiers_emitted": False, "participant_arrays_exported": False},
        "numeric_summary": {key: {"shape": list(value.shape), "dtype": str(value.dtype),
                                  "all_finite": bool(np.isfinite(value).all()),
                                  "nonfinite_count": int((~np.isfinite(value)).sum()),
                                  "min": float(np.min(value)), "max": float(np.max(value))}
                            for key, value in numeric.items()},
        "cohort_structure": {"source_sessions": len(strings["sessions"]),
                             "unique_sessions": len(np.unique(strings["sessions"])),
                             "unique_subjects": len(unique_subjects),
                             "sessions_per_subject_distribution": {str(int(n)): int(np.sum(counts == n)) for n in np.unique(counts)},
                             "earliest_session_rows": len(keep),
                             "matching_existing_cnn_rows": len(rows)},
        "alignment": {
            "earliest_selected_sessions_equal_cnn_in_order": equal(strings["sessions"][keep], cnn["sessions"]),
            "session_lookup_indices_equal_earliest_selection": equal(rows, keep),
            "selected_labels_equal_cnn": equal(numeric["label"][rows], cnn["label"]),
            "selected_scanner_strings_equal_cnn_decoded_site": equal(strings["scanner"][rows], cnn["scanners"][cnn["site"]]),
            "cdr_sessions_equal_cnn": equal(cdr["sessions"], cnn["sessions"]),
            "cdr_labels_equal_selected_features": equal(cdr["label"], numeric["label"][rows]),
            "cdr_scanners_equal_selected_features": equal(cdr["scanner"], strings["scanner"][rows]),
            "age_equal_cdr": equal(cdr["age"], numeric["age"][rows]),
            "age_max_abs_difference": float(np.max(np.abs(cdr["age"]-numeric["age"][rows]))),
            "sex_male1_equal_cdr": equal(cdr["sex_male1"], np.asarray([s.strip().lower() in {"m", "male"} for s in strings["sex"][rows]], dtype=int)),
            "label_equals_cdr_positive": equal(numeric["label"][rows], (cdr["cdr_raw"] > 0).astype(int)),
            "selected_subjects_unique": len(np.unique(strings["subjects"][rows])) == len(rows),
            "row_rule": "Keep earliest _d<integer> session for each subject, sort retained original row indices; matches existing CNN order exactly if checks above are true. Alternatively map root CNN sessions to full source sessions in memory.",
            "row_indices_exported": False,
        },
        "aligned_features": {"shape": list(data["X"].shape), "all_finite": bool(np.isfinite(data["X"]).all()),
                             "matrix_rank": int(np.linalg.matrix_rank(data["X"].astype(float))),
                             "zero_variance_columns": int(np.sum(np.std(data["X"],axis=0)==0)),
                             "label_counts": np.bincount(data["y"]).tolist(),
                             "scanner_counts": np.bincount(data["site"]).tolist()},
        "reference_files": {"cnn_path": str(CNN), "cnn_sha256": hashlib.sha256(CNN.read_bytes()).hexdigest(),
                            "clinical_archive": str(KIT), "clinical_member": CDR_MEMBER,
                            "clinical_member_sha256": hashlib.sha256(data["cdr_bytes"]).hexdigest()},
        "training_or_outcome_metrics_computed": False,
    }


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
