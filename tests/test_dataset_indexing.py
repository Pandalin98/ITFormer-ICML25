import importlib
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _load_dataset_module():
    """Import the dataset module without loading optional model/HDF5 packages."""
    fake_h5py = types.ModuleType("h5py")
    fake_h5py.File = object
    fake_tlm = types.ModuleType("models.TimeLanguageModel")
    fake_tlm.TLMConfig = object
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.PretrainedConfig = object
    fake_transformers.AutoTokenizer = object
    fake_transformers.AutoProcessor = object
    stubs = {
        "h5py": fake_h5py,
        "models.TimeLanguageModel": fake_tlm,
        "transformers": fake_transformers,
    }
    missing = object()
    previous = {name: sys.modules.get(name, missing) for name in stubs}
    sys.modules.update(stubs)
    try:
        dataset_path = Path(__file__).resolve().parents[1] / "dataset" / "dataset.py"
        spec = importlib.util.spec_from_file_location(
            "_issue26_dataset_under_test", dataset_path
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        for name, module in previous.items():
            if module is missing:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


DATASET_MODULE = _load_dataset_module()
TsQaDataset = DATASET_MODULE.TsQaDataset
DataCollator = DATASET_MODULE.DataCollator


def _conversation(stage, question):
    return [
        {
            "stage": str(stage),
            "attribute": f"form-{question}",
            "value": question,
        },
        {"value": f"answer-{question}"},
    ]


class DatasetIndexingTest(unittest.TestCase):
    def _build_dataset(self, rows, shuffle=False):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        path = Path(temp_dir.name) / "qa.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")

        dataset = TsQaDataset.__new__(TsQaDataset)
        dataset.data_path = str(path)
        dataset.shuffle = shuffle
        dataset.h5_file = None
        dataset._build_index()
        return dataset

    def test_flattens_each_qa_to_a_unique_index_and_retains_source_line(self):
        rows = [
            {
                "id": "1",
                "conversations": (
                    _conversation(3, "q0")
                    + _conversation(3, "q1")
                    + _conversation(4, "q2")
                ),
            },
            {
                "id": "2",
                "conversations": (
                    _conversation(1, "q3")
                    + _conversation(5, "ignored")
                    + _conversation(2, "q4")
                ),
            },
        ]
        dataset = self._build_dataset(rows)

        self.assertEqual([item["sample_index"] for item in dataset.datas], list(range(5)))
        self.assertEqual([item["source_line"] for item in dataset.datas], [0, 0, 0, 1, 1])
        self.assertEqual([item["stage"] for item in dataset.datas], [3, 3, 4, 1, 2])

    def test_shuffle_changes_order_without_changing_qa_identity(self):
        rows = [
            {
                "id": "1",
                "conversations": (
                    _conversation(3, "q0")
                    + _conversation(3, "q1")
                    + _conversation(4, "q2")
                ),
            },
            {
                "id": "2",
                "conversations": (
                    _conversation(1, "q3")
                    + _conversation(2, "q4")
                ),
            },
        ]
        with mock.patch.object(
            DATASET_MODULE.random,
            "shuffle",
            side_effect=lambda values: values.reverse(),
        ):
            dataset = self._build_dataset(rows, shuffle=True)

        self.assertEqual([item["sample_index"] for item in dataset.datas], [4, 3, 2, 1, 0])
        source_by_index = {
            item["sample_index"]: item["source_line"] for item in dataset.datas
        }
        self.assertEqual(source_by_index, {0: 0, 1: 0, 2: 0, 3: 1, 4: 1})

    def test_preprocessing_fallback_preserves_unique_qa_identity(self):
        dataset = TsQaDataset.__new__(TsQaDataset)
        dataset.h5_file = None
        dataset.datas = [
            {
                "id": "1",
                "sample_index": 11,
                "source_line": 7,
                "stage": 4,
                "form": "maintenance",
            }
        ]
        dataset.tokenizer = types.SimpleNamespace(eos_token_id=99)
        dataset.config = types.SimpleNamespace(input_len=600, channel_num=33)
        dataset._get_h5_file = mock.Mock(side_effect=RuntimeError("broken HDF5"))

        sample = dataset[0]

        self.assertEqual(sample["index"], 11)
        self.assertEqual(sample["source_line"], 7)
        self.assertEqual(sample["stage"], 4)
        self.assertEqual(sample["form"], "maintenance")
        self.assertEqual(tuple(sample["ts_values"].shape), (600, 33))

    def test_preprocessing_fallback_stacks_with_a_real_sample(self):
        tokenizer = types.SimpleNamespace(
            eos_token_id=99,
            pad_token_id=0,
            padding_side="left",
        )
        dataset = TsQaDataset.__new__(TsQaDataset)
        dataset.h5_file = None
        dataset.tokenizer = tokenizer
        dataset.config = types.SimpleNamespace(input_len=600, channel_num=33)
        dataset._get_h5_file = mock.Mock(side_effect=RuntimeError("broken HDF5"))

        fallback = dataset._get_safe_default_sample(sample_index=8)
        real = {
            **fallback,
            "index": 9,
            "ts_values": torch.ones((600, 33), dtype=torch.float),
        }
        batch = DataCollator(tokenizer)([fallback, real])

        self.assertEqual(tuple(batch["ts_values"].shape), (2, 600, 33))
        self.assertEqual(batch["index"].tolist(), [8, 9])

    def test_strict_preprocessing_raises_instead_of_creating_fake_eval_sample(self):
        dataset = TsQaDataset.__new__(TsQaDataset)
        dataset.h5_file = None
        dataset.strict_preprocessing = True
        dataset.datas = [
            {
                "id": "1",
                "sample_index": 11,
                "source_line": 7,
                "stage": 3,
                "form": "close",
            }
        ]
        dataset.tokenizer = types.SimpleNamespace(eos_token_id=99)
        dataset._get_h5_file = mock.Mock(side_effect=RuntimeError("broken HDF5"))

        with self.assertRaisesRegex(
            RuntimeError,
            "Failed to preprocess QA sample 0",
        ):
            dataset[0]


if __name__ == "__main__":
    unittest.main()
