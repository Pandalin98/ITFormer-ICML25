import re
from typing import Dict, FrozenSet, List, Optional
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from sklearn.metrics import accuracy_score, f1_score
from difflib import SequenceMatcher

def compute_bleu_from_ids(predictions, references):
    """
    Compute BLEU score using str.
    Args:
        predictions (List[str]): Model predicted texts.
        references (List[str]): Reference texts.

    Returns:
        float: BLEU score.
    """
    # Ensure the reference format matches the requirements of corpus_bleu
    predictions = [pred.split() for pred in predictions]
    references = [[ref.split()] for ref in references]
    smooth = SmoothingFunction().method1
    bleu_score = corpus_bleu(references, predictions, smoothing_function=smooth)
    return bleu_score


def compute_rouge_from_ids(predictions, references):
    """
    Compute ROUGE scores using text.
    Args:
        predictions (List[str]): Model predicted texts.
        references (List[str]): Reference texts.

    Returns:
        Dict[str, float]: Contains ROUGE-1, ROUGE-2, and ROUGE-L scores.
    """
    from rouge_score import rouge_scorer

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
    rouge_scores = {"rouge1": 0, "rouge2": 0, "rougeL": 0}
    count = len(predictions)

    for pred, ref in zip(predictions, references):
        score = scorer.score(pred, ref)
        rouge_scores["rouge1"] += score["rouge1"].fmeasure
        rouge_scores["rouge2"] += score["rouge2"].fmeasure
        rouge_scores["rougeL"] += score["rougeL"].fmeasure

    # Average scores
    return {k: v / count for k, v in rouge_scores.items()}




def open_question_metrics(predictions, references, special_ids=[151643]):
    """
    Compute BLEU and ROUGE scores for open-ended questions.
    Args:
        predictions (List[str]): Model predicted texts.
        references (List[str]): Reference texts.
        special_ids (int): Indices used for padding.

    Returns:
        Dict[str, float]: Contains BLEU and ROUGE scores.
    """
    # Remove padding
    decoded_predictions = []
    decoded_labels = []

    for pred, label in zip(predictions, references):
        pred = [token for token in pred if token not in special_ids]
        label = [token for token in label if token not in special_ids]
        decoded_predictions.append(pred)
        decoded_labels.append(label)

    # Compute BLEU
    bleu_score = compute_bleu_from_ids(predictions, references)

    # Compute ROUGE
    rouge_scores = compute_rouge_from_ids(predictions, references)

    return {"BLEU": bleu_score, **rouge_scores}

def compute_rul(predictions, references):
    """
    Compute RUL (Remaining Useful Life) scores.

    Args:
        predictions (List[str]): Model predicted values.
        references (List[str]): Reference values.

    Returns:
        Dict[str, float]: Contains MAE and RMSE scores.
    """
    # Convert strings to numeric values
    predictions = [float(pred) if pred.replace('.', '', 1).isdigit() else 30 for pred in predictions]
    references = [float(ref) for ref in references]

    # Compute MAE (Mean Absolute Error)
    mae = sum(abs(p - r) for p, r in zip(predictions, references)) / len(predictions)

    # Compute RMSE (Root Mean Squared Error)
    mse = sum((p - r) ** 2 for p, r in zip(predictions, references)) / len(predictions)
    rmse = mse ** 0.5

    return {"MAE": mae, "RMSE": rmse, "MSE": mse}




_CLOSED_QUESTION_OPTIONS = frozenset("abcdef")
_BARE_OPTION_SEPARATORS = re.compile(r"[,;/&+]")
_EXPLICIT_OPTION_LIST = re.compile(
    r"\s*[a-f]\s*[\).]"
    r"(?:\s*(?:(?:[,;/&+]|\band\b)\s*)?[a-f]\s*[\).])*\s*",
    re.IGNORECASE,
)
_LEADING_EXPLICIT_OPTION = re.compile(
    r"\s*([a-f])\s*[\).](?:\s+(.*))?\s*",
    re.IGNORECASE | re.DOTALL,
)
_EXPLICIT_OPTION_MARKER = re.compile(
    r"(?<![A-Za-z0-9])([a-f])\s*[\).]",
    re.IGNORECASE,
)


def _parse_bare_options(text: str) -> Optional[FrozenSet[str]]:
    """Parse an answer made only of bare A-F option labels."""
    normalized = _BARE_OPTION_SEPARATORS.sub(" ", text)
    tokens = normalized.split()
    if not tokens:
        return None

    normalized_tokens = [token.lower() for token in tokens]
    if any(
        len(token) != 1 or token not in _CLOSED_QUESTION_OPTIONS
        for token in normalized_tokens
    ):
        return None
    return frozenset(normalized_tokens)


def parse_closed_question_options(value) -> Optional[FrozenSet[str]]:
    """
    Parse strict multiple-choice output.

    Accepted forms are bare option labels (for example ``a`` or ``a c``), a
    list of explicitly punctuated labels (for example ``A), C)``), or one
    leading explicit label followed by its answer text (for example
    ``A) bearing failure``). Free-form text and prose containing conflicting
    explicit labels are intentionally rejected.
    """
    if not isinstance(value, str):
        return None

    text = value.strip()
    if not text:
        return None

    bare_options = _parse_bare_options(text)
    if bare_options is not None:
        return bare_options

    if _EXPLICIT_OPTION_LIST.fullmatch(text):
        return frozenset(
            match.group(1).lower()
            for match in _EXPLICIT_OPTION_MARKER.finditer(text)
        )

    leading_match = _LEADING_EXPLICIT_OPTION.fullmatch(text)
    if leading_match is None:
        return None

    leading_option = leading_match.group(1).lower()
    explicit_options = {
        match.group(1).lower()
        for match in _EXPLICIT_OPTION_MARKER.finditer(text)
    }
    if explicit_options != {leading_option}:
        return None

    trailing_text = leading_match.group(2)
    if trailing_text:
        trailing_bare_options = _parse_bare_options(trailing_text)
        if trailing_bare_options and trailing_bare_options != {leading_option}:
            return None

    return frozenset({leading_option})


def closed_question_metrics(predictions, references, special_id=[151643]):
    """
    Compute evaluation metrics for multiple-choice questions: precision, recall, F1 score, and exact match accuracy.

    Args:
        predictions (List[str]): Model predicted answers in a supported strict
            option format. Unparseable predictions are counted as invalid and
            incorrect.
        references (List[str]): Correct answers in a supported strict option
            format. Empty or unparseable references raise ``ValueError``.

    Returns:
        dict: Contains precision, recall, F1, exact match accuracy, invalid
            prediction count, and invalid prediction rate.
    """
    if len(predictions) != len(references):
        raise ValueError(
            "predictions and references must contain the same number of items"
        )
    if not references:
        raise ValueError("references must not be empty")

    tp, fp, fn = 0, 0, 0
    exact_match_count = 0
    invalid_count = 0

    for index, (pred, ref) in enumerate(zip(predictions, references)):
        ref_set = parse_closed_question_options(ref)
        if ref_set is None:
            raise ValueError(
                f"reference at index {index} is empty or cannot be parsed "
                "as an A-F option answer"
            )

        parsed_prediction = parse_closed_question_options(pred)
        if parsed_prediction is None:
            invalid_count += 1
            pred_set = frozenset()
        else:
            pred_set = parsed_prediction

        # Compute True Positives, False Positives, False Negatives
        tp += len(pred_set & ref_set)  # Correctly predicted options
        fp += len(pred_set - ref_set)  # Incorrectly predicted options
        fn += len(ref_set - pred_set)  # Missed correct options

        # Exact match check
        if pred_set == ref_set:
            exact_match_count += 1

    # Compute metrics
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    exact_match_accuracy = exact_match_count / len(references) if len(references) > 0 else 0.0
    invalid_rate = invalid_count / len(references)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match_accuracy": exact_match_accuracy,
        "invalid_count": invalid_count,
        "invalid_rate": invalid_rate,
    }

# # Example data
# predictions = ['a', 'a token', 'a', 'a', 'b', 'b', 'a b e', 'b', 'a', 'a', 'a', 'b']
# references = ['a', 'a', 'a', 'c', 'b', 'b', 'a b', 'b', 'a', 'a', 'a', 'b']

# # Call function
# metrics = closed_question_metrics(predictions, references)
# print(metrics)
