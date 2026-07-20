import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import Trainer

from EXP.exp_instruct import (
    Exp_Instruct,
    build_eval_loop_output,
    gather_and_merge_eval_results,
)
from utils.result_utils import dataset_sample_indices


def result(index, stage, prediction, label, source_line=0):
    return {
        "index": index,
        "source_line": source_line,
        "stage": stage,
        "prediction": prediction,
        "label": label,
    }


def distributed_gather_worker(rank, world_size, init_file, output_dir):
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        shared = result(2, 4, "shared", "label-two")
        local = (
            [result(0, 1, "rank0-zero", "label-zero"), shared]
            if rank == 0
            else [result(1, 2, "rank1-one", "label-one"), dict(shared)]
        )
        merged = gather_and_merge_eval_results(
            local,
            expected_num_samples=3,
        )
        Path(output_dir, f"rank{rank}.json").write_text(
            json.dumps([item["index"] for item in merged]),
            encoding="utf-8",
        )
    finally:
        dist.destroy_process_group()


class FakeDistributed:
    def __init__(self, remote_results):
        self.remote_results = remote_results
        self.gather_calls = 0

    @staticmethod
    def is_available():
        return True

    @staticmethod
    def is_initialized():
        return True

    @staticmethod
    def get_world_size():
        return 2

    def all_gather_object(self, output, local_results):
        self.gather_calls += 1
        output[0] = local_results
        output[1] = self.remote_results


class DistributedEvalAggregationTest(unittest.TestCase):
    def test_eval_only_checkpoint_is_prepared_after_model_replacement(self):
        loaded_model = mock.Mock()
        loaded_model.cuda.return_value = loaded_model
        loaded_model.tokenizer = types.SimpleNamespace(pad_token_id=7)

        fake_tlm = mock.Mock()
        fake_tlm.from_pretrained.return_value = loaded_model
        model_module = types.ModuleType("models.TimeLanguageModel")
        model_module.TLM = fake_tlm

        prepared_model = mock.Mock()
        accelerator = mock.Mock()
        accelerator.prepare_model.return_value = prepared_model
        accelerator.unwrap_model.return_value = loaded_model

        trainer = Exp_Instruct.__new__(Exp_Instruct)
        trainer.tlmconfig = object()
        trainer.tlmargs = object()
        trainer.accelerator = accelerator

        with mock.patch.dict(
            sys.modules,
            {"models.TimeLanguageModel": model_module},
        ):
            trainer.load_model("checkpoint")

        fake_tlm.from_pretrained.assert_called_once_with(
            "checkpoint",
            config=trainer.tlmconfig,
            ts_config=trainer.tlmargs,
        )
        accelerator.prepare_model.assert_called_once_with(
            loaded_model,
            evaluation_mode=True,
        )
        self.assertIs(trainer.model, prepared_model)
        self.assertIs(trainer.model_wrapped, prepared_model)
        self.assertIs(trainer.processor, loaded_model.tokenizer)
        self.assertEqual(trainer.padding_idx, 7)

    @unittest.skipUnless(
        dist.is_available() and dist.is_gloo_available(),
        "torch.distributed gloo backend is required",
    )
    def test_real_two_process_gather_collects_both_ranks(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            init_file = str(Path(temp_dir, "process-group"))
            mp.spawn(
                distributed_gather_worker,
                args=(2, init_file, temp_dir),
                nprocs=2,
                join=True,
            )

            self.assertEqual(
                json.loads(Path(temp_dir, "rank0.json").read_text(encoding="utf-8")),
                [0, 1, 2],
            )
            self.assertEqual(
                json.loads(Path(temp_dir, "rank1.json").read_text(encoding="utf-8")),
                [0, 1, 2],
            )

    def test_gathers_all_ranks_and_sorts_by_flattened_qa_index(self):
        local = [
            result(0, 1, "rank0-zero", "label-zero"),
            result(2, 4, "shared", "label-two"),
        ]
        remote = [
            result(1, 2, "rank1-one", "label-one"),
            result(2, 4, "shared", "label-two"),
        ]
        distributed = FakeDistributed(remote)

        merged = gather_and_merge_eval_results(
            local,
            expected_num_samples=3,
            distributed=distributed,
        )

        self.assertEqual(distributed.gather_calls, 1)
        self.assertEqual([item["index"] for item in merged], [0, 1, 2])
        self.assertEqual(merged[1]["prediction"], "rank1-one")

    def test_deduplicates_flattened_index_not_source_line(self):
        local = [
            result(10, 3, "a", "a", source_line=7),
            result(11, 4, "repair", "repair", source_line=7),
        ]
        remote = [dict(local[1])]

        merged = gather_and_merge_eval_results(
            local,
            expected_num_samples=2,
            distributed=FakeDistributed(remote),
        )

        self.assertEqual([item["index"] for item in merged], [10, 11])
        self.assertEqual([item["stage"] for item in merged], [3, 4])

    def test_fails_loudly_if_unique_results_do_not_cover_eval_dataset(self):
        with self.assertRaisesRegex(RuntimeError, "expected 2"):
            gather_and_merge_eval_results(
                [result(0, 1, "only", "only")],
                expected_num_samples=2,
                distributed=FakeDistributed([]),
            )

    def test_fails_if_count_matches_but_index_set_is_wrong(self):
        with self.assertRaisesRegex(RuntimeError, "wrong QA index set"):
            gather_and_merge_eval_results(
                [
                    result(0, 1, "zero", "zero"),
                    result(99, 2, "wrong", "wrong"),
                ],
                expected_num_samples=2,
                expected_indices=[0, 1],
                distributed=FakeDistributed([]),
            )

    def test_subset_expected_indices_follow_flattened_qa_identity(self):
        base = types.SimpleNamespace(
            datas=[
                {"sample_index": 10},
                {"sample_index": 20},
                {"sample_index": 30},
            ]
        )
        subset = types.SimpleNamespace(dataset=base, indices=[2, 0])

        self.assertEqual(dataset_sample_indices(subset), [30, 10])

    def test_eval_loop_uses_inference_result_fields_as_metric_inputs(self):
        merged = [
            result(0, 1, "open prediction", "open label"),
            result(1, 2, "b", "a"),
            result(2, 3, "c", "c"),
            result(3, 4, "action prediction", "action label"),
        ]
        output = build_eval_loop_output(merged, num_samples=len(merged))

        self.assertEqual(output.predictions, [item["prediction"] for item in merged])
        self.assertEqual(output.label_ids, [item["label"] for item in merged])
        self.assertEqual(output.pred_extra["stages"], [item["stage"] for item in merged])
        self.assertEqual(output.pred_extra["indices"], [item["index"] for item in merged])

        calls = []

        def open_metrics(predictions, labels, special_ids):
            calls.append(("open", predictions, labels, special_ids))
            return {"score": len(predictions)}

        def closed_metrics(predictions, labels, special_ids):
            calls.append(("closed", predictions, labels, special_ids))
            return {"score": len(predictions)}

        metrics_module = types.ModuleType("utils.metrics")
        metrics_module.open_question_metrics = open_metrics
        metrics_module.closed_question_metrics = closed_metrics

        trainer = Exp_Instruct.__new__(Exp_Instruct)
        trainer.special_id = [99]
        with mock.patch.dict(sys.modules, {"utils.metrics": metrics_module}):
            metrics = trainer.custom_compute_metrics(output)

        self.assertEqual(
            calls,
            [
                ("open", ["open prediction"], ["open label"], [99]),
                ("closed", ["b"], ["a"], [99]),
                ("closed", ["c"], ["c"], [99]),
                ("open", ["action prediction"], ["action label"], [99]),
            ],
        )
        self.assertEqual(
            metrics,
            {
                "stage1_score": 1,
                "stage2_score": 1,
                "stage3_score": 1,
                "stage4_score": 1,
            },
        )

    def test_eval_dataloader_disables_drop_last_without_changing_training(self):
        trainer = Exp_Instruct.__new__(Exp_Instruct)
        trainer.args = types.SimpleNamespace(dataloader_drop_last=True)
        observed = []

        def parent_get_eval_dataloader(self, eval_dataset=None):
            observed.append(self.args.dataloader_drop_last)
            return "prepared-eval-loader"

        with mock.patch.object(
            Trainer,
            "get_eval_dataloader",
            parent_get_eval_dataloader,
        ):
            loader = trainer.get_eval_dataloader("eval-dataset")

        self.assertEqual(loader, "prepared-eval-loader")
        self.assertEqual(observed, [False])
        self.assertTrue(trainer.args.dataloader_drop_last)


if __name__ == "__main__":
    unittest.main()
