import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.metrics import closed_question_metrics


class ClosedQuestionMetricsTest(unittest.TestCase):
    def test_preserves_existing_lowercase_a_to_e_behavior(self):
        predictions = ["a", "b", "a b e", "d", "e"]
        references = ["a", "c", "a b", "d", "a e"]

        metrics = closed_question_metrics(predictions, references)

        self.assertAlmostEqual(metrics["precision"], 5 / 7)
        self.assertAlmostEqual(metrics["recall"], 5 / 7)
        self.assertAlmostEqual(metrics["f1"], 5 / 7)
        self.assertAlmostEqual(metrics["exact_match_accuracy"], 2 / 5)
        self.assertEqual(metrics["invalid_count"], 0)
        self.assertEqual(metrics["invalid_rate"], 0.0)

    def test_accepts_bare_and_explicit_single_or_multiple_options(self):
        predictions = [
            "A",
            "B)",
            "c.",
            "A C",
            "A), C)",
            "D) bearing failure",
            "E. normal operation",
            "F",
        ]
        references = ["a", "b", "c", "a c", "a c", "d", "e", "f"]

        metrics = closed_question_metrics(predictions, references)

        self.assertEqual(
            metrics,
            {
                "precision": 1.0,
                "recall": 1.0,
                "f1": 1.0,
                "exact_match_accuracy": 1.0,
                "invalid_count": 0,
                "invalid_rate": 0.0,
            },
        )

    def test_unparseable_and_conflicting_predictions_are_invalid_and_incorrect(self):
        predictions = [
            "",
            "a token",
            "the answer is A",
            "A or B",
            "A) first answer; B) second answer",
            "G",
        ]
        references = ["a", "a", "a", "a", "a", "a"]

        metrics = closed_question_metrics(predictions, references)

        self.assertEqual(metrics["precision"], 0.0)
        self.assertEqual(metrics["recall"], 0.0)
        self.assertEqual(metrics["f1"], 0.0)
        self.assertEqual(metrics["exact_match_accuracy"], 0.0)
        self.assertEqual(metrics["invalid_count"], 6)
        self.assertEqual(metrics["invalid_rate"], 1.0)

    def test_invalid_prediction_counts_as_a_missed_reference(self):
        metrics = closed_question_metrics(["", "b"], ["a", "b"])

        self.assertEqual(metrics["precision"], 1.0)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertAlmostEqual(metrics["f1"], 2 / 3)
        self.assertEqual(metrics["exact_match_accuracy"], 0.5)
        self.assertEqual(metrics["invalid_count"], 1)
        self.assertEqual(metrics["invalid_rate"], 0.5)

    def test_empty_or_unparseable_reference_raises(self):
        for reference in ("", "   ", "answer is A", "G", None):
            with self.subTest(reference=reference):
                with self.assertRaisesRegex(ValueError, "reference at index 0"):
                    closed_question_metrics(["a"], [reference])

    def test_empty_reference_collection_raises(self):
        with self.assertRaisesRegex(ValueError, "references must not be empty"):
            closed_question_metrics([], [])

    def test_length_mismatch_raises_instead_of_silently_truncating(self):
        with self.assertRaisesRegex(ValueError, "same number of items"):
            closed_question_metrics(["a"], ["a", "b"])


if __name__ == "__main__":
    unittest.main()
