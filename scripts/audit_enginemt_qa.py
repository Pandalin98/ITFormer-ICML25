#!/usr/bin/env python
"""Audit EngineMT-QA HDF5 structure and QA-to-row alignment."""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence


class MissingDependencyError(RuntimeError):
    """Raised when an optional dependency required by the audit is absent."""


def _load_h5py():
    try:
        import h5py
    except ImportError as exc:
        raise MissingDependencyError(
            "Auditing --h5 requires the optional dependency 'h5py'. "
            "Install it with `python -m pip install h5py`, then rerun the audit."
        ) from exc
    return h5py


def _json_safe(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        if len(value) > 50:
            return {
                "length": len(value),
                "first_values": [_json_safe(item) for item in value[:10]],
            }
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def _attrs_to_dict(attrs: Any) -> dict[str, Any]:
    return {str(key): _json_safe(value) for key, value in attrs.items()}


def _normalize_ids(raw_ids: Any, line_number: int) -> tuple[list[int], list[str]]:
    errors: list[str] = []
    values = raw_ids if isinstance(raw_ids, list) else [raw_ids]
    normalized: list[int] = []

    for value in values:
        if isinstance(value, bool):
            errors.append(f"QA line {line_number}: boolean ID {value!r} is invalid")
            continue
        try:
            normalized_id = int(value)
        except (TypeError, ValueError):
            errors.append(f"QA line {line_number}: ID {value!r} is not an integer")
            continue
        normalized.append(normalized_id)

    return normalized, errors


def audit_qa(
    path: str | Path, expected_list_id_count: int = 10
) -> tuple[dict[str, Any], set[int], list[str]]:
    stages: collections.Counter[int] = collections.Counter()
    list_lengths: collections.Counter[int] = collections.Counter()
    ids: list[int] = []
    errors: list[str] = []
    line_count = 0
    scalar_id_records = 0
    list_id_records = 0

    with Path(path).open(encoding="utf-8") as handle:
        for line_count, line in enumerate(handle, start=1):
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                errors.append(f"QA line {line_count}: invalid JSON ({exc.msg})")
                continue

            raw_ids = item.get("id")
            if isinstance(raw_ids, list):
                list_id_records += 1
                list_lengths[len(raw_ids)] += 1
                if len(raw_ids) != expected_list_id_count:
                    errors.append(
                        f"QA line {line_count}: list ID count {len(raw_ids)} "
                        f"does not match expected {expected_list_id_count}"
                    )
            else:
                scalar_id_records += 1

            normalized_ids, id_errors = _normalize_ids(raw_ids, line_count)
            ids.extend(normalized_ids)
            errors.extend(id_errors)

            conversations = item.get("conversations")
            if not isinstance(conversations, list):
                errors.append(f"QA line {line_count}: conversations is not a list")
                continue
            if len(conversations) % 2:
                errors.append(
                    f"QA line {line_count}: conversations has odd length "
                    f"{len(conversations)}"
                )

            for offset in range(0, len(conversations), 2):
                turn = conversations[offset]
                if not isinstance(turn, dict):
                    errors.append(
                        f"QA line {line_count}: conversation {offset} is not an object"
                    )
                    continue
                stage = turn.get("stage")
                if stage in {"1", "2", "3", "4"}:
                    stages[int(stage)] += 1

    unique_ids = set(ids)
    report = {
        "path": str(path),
        "jsonl_lines": line_count,
        "qa_total": sum(stages.values()),
        "stages": {str(stage): stages[stage] for stage in sorted(stages)},
        "scalar_id_records": scalar_id_records,
        "list_id_records": list_id_records,
        "list_id_lengths": {
            str(length): count for length, count in sorted(list_lengths.items())
        },
        "id_reference_count": len(ids),
        "unique_id_count": len(unique_ids),
        "id_min": min(unique_ids) if unique_ids else None,
        "id_max": max(unique_ids) if unique_ids else None,
    }
    return report, unique_ids, errors


def validate_alignment(
    seq_shape: Sequence[int],
    data_ids: Sequence[int] | None,
    qa_ids: Iterable[int],
    *,
    expected_sequence_length: int = 600,
    expected_channels: int = 33,
) -> tuple[dict[str, Any], list[str]]:
    shape = tuple(int(dimension) for dimension in seq_shape)
    qa_id_set = set(qa_ids)
    errors: list[str] = []
    details: dict[str, Any] = {
        "seq_data_shape": list(shape),
        "expected_sequence_length": expected_sequence_length,
        "expected_channels": expected_channels,
    }

    if len(shape) != 3:
        errors.append(f"seq_data shape {shape} is invalid: expected rank 3")
        return details, errors

    row_count, sequence_length, channel_count = shape
    details.update(
        {
            "row_count": row_count,
            "sequence_length": sequence_length,
            "channel_count": channel_count,
        }
    )
    if sequence_length != expected_sequence_length:
        errors.append(
            f"seq_data sequence length {sequence_length} does not match "
            f"expected {expected_sequence_length}"
        )
    if channel_count != expected_channels:
        errors.append(
            f"seq_data channel count {channel_count} does not match "
            f"expected {expected_channels}"
        )

    out_of_range = sorted(
        qa_id for qa_id in qa_id_set if qa_id < 1 or qa_id > row_count
    )
    details["qa_id_expected_range"] = [1, row_count]
    details["qa_id_out_of_range_count"] = len(out_of_range)
    details["qa_id_out_of_range_examples"] = out_of_range[:20]
    if out_of_range:
        errors.append(
            f"{len(out_of_range)} unique QA IDs are outside [1, {row_count}]; "
            f"examples: {out_of_range[:10]}"
        )

    if data_ids is None:
        errors.append("HDF5 dataset 'data_ID' is missing; row alignment is unverified")
        return details, errors

    normalized_data_ids = [int(value) for value in data_ids]
    details.update(
        {
            "data_id_count": len(normalized_data_ids),
            "data_id_min": min(normalized_data_ids) if normalized_data_ids else None,
            "data_id_max": max(normalized_data_ids) if normalized_data_ids else None,
            "data_id_unique_count": len(set(normalized_data_ids)),
        }
    )
    if len(normalized_data_ids) != row_count:
        errors.append(
            f"data_ID length {len(normalized_data_ids)} does not match "
            f"seq_data row count {row_count}"
        )

    expected_data_ids = list(range(1, row_count + 1))
    data_id_is_one_based_row_order = normalized_data_ids == expected_data_ids
    details["data_id_is_one_based_row_order"] = data_id_is_one_based_row_order
    if not data_id_is_one_based_row_order:
        mismatch_examples = []
        for row_index, (actual, expected) in enumerate(
            zip(normalized_data_ids, expected_data_ids)
        ):
            if actual != expected:
                mismatch_examples.append(
                    {
                        "row_index": row_index,
                        "data_ID": actual,
                        "expected_data_ID": expected,
                    }
                )
                if len(mismatch_examples) == 10:
                    break
        details["data_id_order_mismatch_examples"] = mismatch_examples
        errors.append(
            "data_ID is not the exact one-based row order 1..len(seq_data)"
        )

    valid_qa_ids = sorted(
        qa_id for qa_id in qa_id_set if 1 <= qa_id <= row_count
    )
    mapping_mismatches = []
    for qa_id in valid_qa_ids:
        row_index = qa_id - 1
        if row_index >= len(normalized_data_ids):
            break
        if normalized_data_ids[row_index] != qa_id:
            mapping_mismatches.append(
                {
                    "qa_id": qa_id,
                    "row_index": row_index,
                    "data_ID": normalized_data_ids[row_index],
                }
            )
            if len(mapping_mismatches) == 20:
                break
    details["qa_id_row_mapping_mismatch_count_at_least"] = len(mapping_mismatches)
    details["qa_id_row_mapping_mismatch_examples"] = mapping_mismatches
    if mapping_mismatches:
        errors.append(
            "QA IDs do not consistently map to seq_data[id - 1]; "
            f"examples: {mapping_mismatches[:5]}"
        )

    return details, errors


def audit_h5(
    path: str | Path,
    qa_ids: Iterable[int],
    *,
    expected_sequence_length: int = 600,
    expected_channels: int = 33,
    h5py_module=None,
) -> tuple[dict[str, Any], list[str]]:
    h5py_module = h5py_module or _load_h5py()

    with h5py_module.File(path, "r") as handle:
        report: dict[str, Any] = {
            "path": str(path),
            "root_keys": sorted(str(key) for key in handle.keys()),
            "root_attrs": _attrs_to_dict(handle.attrs),
            "datasets": {},
        }

        for name in handle.keys():
            obj = handle[name]
            report["datasets"][str(name)] = {
                "shape": list(obj.shape),
                "dtype": str(obj.dtype),
                "attrs": _attrs_to_dict(obj.attrs),
            }

        if "seq_data" not in handle:
            return report, ["HDF5 dataset 'seq_data' is missing"]

        seq_shape = handle["seq_data"].shape
        data_ids = handle["data_ID"][:] if "data_ID" in handle else None
        alignment, errors = validate_alignment(
            seq_shape,
            data_ids,
            qa_ids,
            expected_sequence_length=expected_sequence_length,
            expected_channels=expected_channels,
        )
        report["alignment"] = alignment
        return report, errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Audit EngineMT-QA JSONL IDs and their one-based alignment with "
            "HDF5 seq_data rows. This reports structural metadata only and "
            "does not infer channel semantics."
        )
    )
    parser.add_argument("--qa", required=True, help="QA JSONL path")
    parser.add_argument("--h5", help="EngineMT-QA HDF5 path")
    parser.add_argument(
        "--expected-sequence-length",
        type=int,
        default=600,
        help="Expected seq_data time dimension (default: 600)",
    )
    parser.add_argument(
        "--expected-channels",
        type=int,
        default=33,
        help="Expected released seq_data channel dimension (default: 33)",
    )
    parser.add_argument(
        "--expected-list-id-count",
        type=int,
        default=10,
        help="Expected number of IDs in multi-cycle QA records (default: 10)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        qa_report, qa_ids, errors = audit_qa(
            args.qa, expected_list_id_count=args.expected_list_id_count
        )
        report: dict[str, Any] = {"qa": qa_report}
        if args.h5:
            h5_report, h5_errors = audit_h5(
                args.h5,
                qa_ids,
                expected_sequence_length=args.expected_sequence_length,
                expected_channels=args.expected_channels,
            )
            report["h5"] = h5_report
            errors.extend(h5_errors)
    except MissingDependencyError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except (OSError, KeyError, TypeError, ValueError) as exc:
        print(f"ERROR: audit could not be completed: {exc}", file=sys.stderr)
        return 2

    report["status"] = "error" if errors else "ok"
    report["errors"] = errors
    print(json.dumps(report, indent=2, sort_keys=True))
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
