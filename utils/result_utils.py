"""Utilities shared by training-time evaluation and standalone inference."""

from typing import Any, Dict, Iterable, List


def _parse_sample_index(result: Dict[str, Any]) -> int:
    try:
        raw_index = result["index"]
        if isinstance(raw_index, bool):
            raise TypeError("boolean indices are not valid QA indices")
        return int(raw_index)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Result has no valid integer QA index: {result!r}") from exc


def _conflicting_keys(previous: Dict[str, Any], current: Dict[str, Any]) -> List[str]:
    """Return payload fields that disagree for the same normalized QA index."""
    keys = (set(previous) | set(current)) - {"index"}
    return sorted(
        key
        for key in keys
        if key not in previous
        or key not in current
        or previous[key] != current[key]
    )


def merge_unique_results(result_groups: Iterable[Iterable[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Merge distributed results and remove sampler-padding duplicates.

    ``index`` must be the flattened QA index, not the source JSONL line number.
    One EngineMT-QA JSONL line contains 3 or 15 QA pairs, so line-number based
    deduplication silently drops most questions (including Stage 4). Duplicate
    indices are accepted only when every non-index field is identical.
    """
    unique: Dict[int, Dict[str, Any]] = {}
    for group in result_groups:
        for result in group:
            sample_index = _parse_sample_index(result)
            normalized_result = dict(result)
            normalized_result["index"] = sample_index
            if sample_index in unique:
                previous = unique[sample_index]
                conflicting_keys = _conflicting_keys(previous, normalized_result)
                if conflicting_keys:
                    raise ValueError(
                        "Conflicting distributed results for QA index "
                        f"{sample_index}; differing fields: {', '.join(conflicting_keys)}"
                    )
                continue
            unique[sample_index] = normalized_result
    return [unique[index] for index in sorted(unique)]


def dataset_sample_indices(dataset) -> List[int]:
    """Read stable flattened QA identities without preprocessing samples."""
    if hasattr(dataset, "indices") and hasattr(dataset, "dataset"):
        parent_indices = dataset_sample_indices(dataset.dataset)
        return [parent_indices[int(position)] for position in dataset.indices]
    if hasattr(dataset, "datas"):
        return [int(item["sample_index"]) for item in dataset.datas]
    return list(range(len(dataset)))


def rank_strided_positions(
    dataset_length: int,
    *,
    process_index: int,
    num_processes: int,
) -> List[int]:
    """Return an exact, padding-free dataset partition for one process.

    Accelerate's default even-batch sharding repeats leading samples when the
    dataset size is not divisible by the global batch geometry. Greedy BF16
    generation can still differ at a token boundary across GPUs, so duplicated
    sampler-padding samples are not a safe inference contract. A strided
    partition covers every dataset position exactly once and never pads.
    """
    if dataset_length < 0:
        raise ValueError("dataset_length must be non-negative")
    if num_processes <= 0:
        raise ValueError("num_processes must be positive")
    if not 0 <= process_index < num_processes:
        raise ValueError("process_index must be in [0, num_processes)")
    return list(range(process_index, dataset_length, num_processes))
