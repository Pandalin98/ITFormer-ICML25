import sys
import unittest
from pathlib import Path

from accelerate.data_loader import BatchSamplerShard
from torch.utils.data import BatchSampler, DistributedSampler, SequentialSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.result_utils import merge_unique_results, rank_strided_positions


class MergeUniqueResultsTest(unittest.TestCase):
    def test_preserves_multiple_qas_from_one_source_line(self):
        groups = [[
            {"index": 0, "source_line": 7, "stage": 3, "prediction": "d", "label": "d"},
            {"index": 1, "source_line": 7, "stage": 3, "prediction": "d", "label": "d"},
            {"index": 2, "source_line": 7, "stage": 4, "prediction": "repair", "label": "repair"},
        ]]
        merged = merge_unique_results(groups)
        self.assertEqual([item["index"] for item in merged], [0, 1, 2])
        self.assertEqual([item["source_line"] for item in merged], [7, 7, 7])
        self.assertEqual([item["stage"] for item in merged], [3, 3, 4])

    def test_removes_only_distributed_padding_duplicate(self):
        item = {"index": 2, "stage": 4, "prediction": "repair", "label": "repair"}
        merged = merge_unique_results([[item], [dict(item)]])
        self.assertEqual(merged, [item])

    def test_single_replica_preserves_all_production_sized_indices(self):
        qa_count = 42477
        sampler = DistributedSampler(
            range(qa_count), num_replicas=1, rank=0, shuffle=False, drop_last=False
        )
        group = [{"index": index} for index in sampler]
        merged = merge_unique_results([group])
        self.assertEqual(len(group), qa_count)
        self.assertEqual(len(merged), qa_count)
        self.assertEqual([merged[0]["index"], merged[-1]["index"]], [0, qa_count - 1])

    def test_two_replicas_remove_exactly_one_distributed_sampler_padding_item(self):
        qa_count = 42477
        groups = [
            [
                {"index": index, "source_line": index // 3}
                for index in DistributedSampler(
                    range(qa_count),
                    num_replicas=2,
                    rank=rank,
                    shuffle=False,
                    drop_last=False,
                )
            ]
            for rank in range(2)
        ]
        merged = merge_unique_results(groups)
        self.assertEqual([len(group) for group in groups], [21239, 21239])
        self.assertEqual(sum(map(len, groups)), 42478)
        self.assertEqual(len(merged), qa_count)
        self.assertEqual([item["index"] for item in merged], list(range(qa_count)))

    def test_two_accelerate_processes_remove_three_even_batch_padding_items(self):
        qa_count = 42477
        base_sampler = SequentialSampler(range(qa_count))
        base_batch_sampler = BatchSampler(base_sampler, batch_size=12, drop_last=False)
        groups = []
        for rank in range(2):
            shard = BatchSamplerShard(
                base_batch_sampler,
                num_processes=2,
                process_index=rank,
                split_batches=False,
                even_batches=True,
            )
            groups.append(
                [{"index": index} for batch in shard for index in batch]
            )

        merged = merge_unique_results(groups)
        self.assertEqual([len(group) for group in groups], [21240, 21240])
        self.assertEqual(sum(map(len, groups)), 42480)
        self.assertEqual(len(merged), qa_count)
        self.assertEqual([item["index"] for item in merged], list(range(qa_count)))

    def test_rejects_any_conflicting_duplicate_payload(self):
        cases = {
            "prediction": (
                {"index": 2, "source_line": 7, "stage": 4, "prediction": "a", "label": "x"},
                {"index": 2, "source_line": 7, "stage": 4, "prediction": "b", "label": "x"},
            ),
            "source_line": (
                {"index": 2, "source_line": 7, "stage": 4, "prediction": "a", "label": "x"},
                {"index": 2, "source_line": 8, "stage": 4, "prediction": "a", "label": "x"},
            ),
            "is_correct": (
                {"index": 2, "stage": 4, "prediction": "a", "label": "x", "is_correct": False},
                {"index": 2, "stage": 4, "prediction": "a", "label": "x", "is_correct": True},
            ),
            "input": (
                {"index": 2, "stage": 4, "input": None},
                {"index": 2, "stage": 4},
            ),
        }
        for field, (left, right) in cases.items():
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, field):
                    merge_unique_results([[left], [right]])

    def test_accepts_equivalent_integer_index_representations(self):
        left = {"index": 2, "stage": 4}
        right = {"index": "2", "stage": 4}
        self.assertEqual(merge_unique_results([[left], [right]]), [left])
        self.assertEqual(right["index"], "2")

    def test_rejects_missing_or_non_integer_index(self):
        for item in ({}, {"index": "not-an-int"}, {"index": True}):
            with self.subTest(item=item):
                with self.assertRaisesRegex(ValueError, "valid integer QA index"):
                    merge_unique_results([[item]])

    def test_padding_free_rank_partition_has_exact_disjoint_coverage(self):
        for qa_count in (0, 1, 2, 25, 42477, 44037):
            with self.subTest(qa_count=qa_count):
                groups = [
                    rank_strided_positions(
                        qa_count,
                        process_index=rank,
                        num_processes=2,
                    )
                    for rank in range(2)
                ]
                self.assertFalse(set(groups[0]) & set(groups[1]))
                self.assertEqual(
                    sorted(groups[0] + groups[1]),
                    list(range(qa_count)),
                )

    def test_padding_free_rank_partition_validates_arguments(self):
        cases = (
            (-1, 0, 2),
            (1, 0, 0),
            (1, -1, 2),
            (1, 2, 2),
        )
        for dataset_length, process_index, num_processes in cases:
            with self.subTest(
                dataset_length=dataset_length,
                process_index=process_index,
                num_processes=num_processes,
            ):
                with self.assertRaises(ValueError):
                    rank_strided_positions(
                        dataset_length,
                        process_index=process_index,
                        num_processes=num_processes,
                    )


if __name__ == "__main__":
    unittest.main()
