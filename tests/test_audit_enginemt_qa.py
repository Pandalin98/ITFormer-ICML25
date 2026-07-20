import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "audit_enginemt_qa.py"
SPEC = importlib.util.spec_from_file_location("audit_enginemt_qa", SCRIPT_PATH)
audit = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(audit)


class FakeDataset:
    def __init__(self, shape, values=None, dtype="float64", attrs=None):
        self.shape = shape
        self.values = values
        self.dtype = dtype
        self.attrs = attrs or {}

    def __getitem__(self, key):
        if isinstance(key, slice):
            return list(self.values)
        return self.values[key]


class FakeFile:
    def __init__(self, datasets, attrs=None):
        self.datasets = datasets
        self.attrs = attrs or {}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def __contains__(self, key):
        return key in self.datasets

    def __getitem__(self, key):
        return self.datasets[key]

    def keys(self):
        return self.datasets.keys()


class FakeH5py:
    def __init__(self, handle):
        self.handle = handle

    def File(self, path, mode):
        self.path = path
        self.mode = mode
        return self.handle


def write_qa(path, ids):
    item = {
        "id": ids,
        "conversations": [
            {"from": "human", "stage": "1", "attribute": "open", "value": "q"},
            {"from": "gpt", "value": "a"},
        ],
    }
    path.write_text(json.dumps(item) + "\n", encoding="utf-8")


class AuditEngineMtQaTests(unittest.TestCase):
    def test_audit_qa_counts_flattened_pairs_and_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            qa_path = Path(tmp) / "qa.jsonl"
            rows = [
                {
                    "id": "2",
                    "conversations": [
                        {
                            "from": "human",
                            "stage": "1",
                            "attribute": "open",
                            "value": "q1",
                        },
                        {"from": "gpt", "value": "a1"},
                        {
                            "from": "human",
                            "stage": "2",
                            "attribute": "close",
                            "value": "q2",
                        },
                        {"from": "gpt", "value": "a2"},
                    ],
                },
                {
                    "id": [str(value) for value in range(1, 11)],
                    "conversations": [
                        {
                            "from": "human",
                            "stage": "3",
                            "attribute": "close",
                            "value": "q3",
                        },
                        {"from": "gpt", "value": "a3"},
                    ],
                },
            ]
            qa_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )

            report, ids, errors = audit.audit_qa(qa_path)

        self.assertEqual(errors, [])
        self.assertEqual(report["jsonl_lines"], 2)
        self.assertEqual(report["qa_total"], 3)
        self.assertEqual(report["stages"], {"1": 1, "2": 1, "3": 1})
        self.assertEqual(report["list_id_lengths"], {"10": 1})
        self.assertEqual(report["id_reference_count"], 11)
        self.assertEqual(ids, set(range(1, 11)))

    def test_validate_alignment_rejects_bad_shape_and_out_of_range_id(self):
        details, errors = audit.validate_alignment(
            (3, 600, 32), [1, 2, 3], {1, 4}
        )

        self.assertEqual(details["channel_count"], 32)
        self.assertEqual(details["qa_id_out_of_range_examples"], [4])
        self.assertTrue(any("channel count 32" in error for error in errors))
        self.assertTrue(any("outside [1, 3]" in error for error in errors))

    def test_validate_alignment_rejects_data_id_row_mismatch(self):
        details, errors = audit.validate_alignment(
            (3, 600, 33), [1, 3, 2], {1, 2, 3}
        )

        self.assertFalse(details["data_id_is_one_based_row_order"])
        self.assertTrue(any("one-based row order" in error for error in errors))
        self.assertTrue(any("seq_data[id - 1]" in error for error in errors))

    def test_audit_h5_reports_shape_attrs_and_valid_alignment(self):
        fake_file = FakeFile(
            {
                "data_ID": FakeDataset((3,), [1, 2, 3], "int64"),
                "seq_data": FakeDataset((3, 600, 33), dtype="float64"),
            }
        )
        report, errors = audit.audit_h5(
            "fake.h5", {1, 3}, h5py_module=FakeH5py(fake_file)
        )

        self.assertEqual(errors, [])
        self.assertEqual(report["root_keys"], ["data_ID", "seq_data"])
        self.assertEqual(report["root_attrs"], {})
        self.assertEqual(report["datasets"]["seq_data"]["shape"], [3, 600, 33])
        self.assertTrue(
            report["alignment"]["data_id_is_one_based_row_order"]
        )

    def test_missing_h5py_error_is_explicit(self):
        real_import = __import__

        def blocked_import(name, *args, **kwargs):
            if name == "h5py":
                raise ModuleNotFoundError("No module named 'h5py'", name="h5py")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=blocked_import):
            with self.assertRaises(audit.MissingDependencyError) as context:
                audit._load_h5py()

        self.assertIn("requires the optional dependency 'h5py'", str(context.exception))
        self.assertIn("python -m pip install h5py", str(context.exception))

    def test_main_returns_nonzero_for_bad_shape_and_id_range(self):
        with tempfile.TemporaryDirectory() as tmp:
            qa_path = Path(tmp) / "qa.jsonl"
            write_qa(qa_path, "4")
            fake_file = FakeFile(
                {
                    "data_ID": FakeDataset((3,), [1, 2, 3], "int64"),
                    "seq_data": FakeDataset((3, 600, 33), dtype="float64"),
                }
            )
            stdout = io.StringIO()
            with mock.patch.object(
                audit, "_load_h5py", return_value=FakeH5py(fake_file)
            ), contextlib.redirect_stdout(stdout):
                exit_code = audit.main(
                    [
                        "--qa",
                        str(qa_path),
                        "--h5",
                        "fake.h5",
                        "--expected-channels",
                        "32",
                    ]
                )

        report = json.loads(stdout.getvalue())
        self.assertEqual(exit_code, 1)
        self.assertEqual(report["status"], "error")
        self.assertTrue(any("channel count 33" in error for error in report["errors"]))
        self.assertTrue(any("outside [1, 3]" in error for error in report["errors"]))

    def test_main_returns_two_when_h5py_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            qa_path = Path(tmp) / "qa.jsonl"
            write_qa(qa_path, "1")
            stderr = io.StringIO()
            with mock.patch.object(
                audit,
                "_load_h5py",
                side_effect=audit.MissingDependencyError(
                    "Auditing --h5 requires the optional dependency 'h5py'."
                ),
            ), contextlib.redirect_stderr(stderr):
                exit_code = audit.main(["--qa", str(qa_path), "--h5", "fake.h5"])

        self.assertEqual(exit_code, 2)
        self.assertIn("ERROR:", stderr.getvalue())
        self.assertIn("requires the optional dependency 'h5py'", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
