from typing import List, Dict
from nltk.translate.bleu_score import corpus_bleu,sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sklearn.metrics import accuracy_score, f1_score
from difflib import SequenceMatcher

def compute_bleu_from_ids(predictions, references):
    """
    Calculate BLEU score using strings.
    Args:
        predictions (List[str]): Model predicted text
        references (List[str]): Reference answer text

    Returns:
        float: BLEU score.
    """
    predictions = [pred.split() for pred in predictions]
    references = [[ref.split()] for ref in references]
    smooth = SmoothingFunction().method1
    bleu_score = corpus_bleu(references, predictions, smoothing_function=smooth)
    return bleu_score


def compute_rouge_from_ids(predictions, references):
    """
    Calculate ROUGE score using text.
    Args:
        predictions (List[str]): Model predicted text.
        references (List[str]): Reference answer text.

    Returns:
        Dict[str, float]: Scores containing ROUGE-1, ROUGE-2, and ROUGE-L.
    """
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




def open_question_metrics(predictions, references,special_ids = [151643]):
    """
    Calculate BLEU and ROUGE scores for open-ended questions.
    Args:
        predictions (List[str]): Model predicted text.
        references (List[str]): Reference answer text.
        special_ids (int): Index used for padding.

    Returns:
        Dict[str, float]: Scores containing BLEU and ROUGE.
    """
    decoded_predictions = []
    decoded_labels = []

    for pred, label in zip(predictions, references):
        pred = [token for token in pred if token not in  special_ids]
        label = [token for token in label if token not in special_ids]
        decoded_predictions.append(pred)
        decoded_labels.append(label)

    bleu_score = compute_bleu_from_ids(predictions, references)

    rouge_scores = compute_rouge_from_ids(predictions, references)

    return {"BLEU": bleu_score, **rouge_scores}

def compute_rul(predictions, references):
    """
    Calculate RUL score.

    Args:
        predictions (List[str]): Numeric values predicted by the model.
        references (List[str]): Reference numeric values.

    Returns:
        Dict[str, float]: Scores containing MAE and RMSE.
    """
    predictions = [float(pred) if pred.replace('.', '', 1).isdigit() else 30 for pred in predictions]
    references = [float(ref) for ref in references]

    mae = sum(abs(p - r) for p, r in zip(predictions, references)) / len(predictions)

    mse = sum((p - r) ** 2 for p, r in zip(predictions, references)) / len(predictions)
    rmse = mse ** 0.5

    return {"MAE": mae, "RMSE": rmse,"MSE":mse}




def closed_question_metrics(predictions, references,special_id=[151643]):
    """
    Calculate evaluation metrics for multiple-choice questions: precision, recall, F1 score, and exact match rate.

    Args:
        predictions (List[str]): Model predicted answers list, single or multiple choices separated by spaces (e.g., 'a b e').
        references (List[str]): Correct answers list, single or multiple choices separated by spaces (e.g., 'a b').

    Returns:
        dict: Results containing precision, recall, F1, and exact match rate.
    """
    tp, fp, fn = 0, 0, 0
    exact_match_count = 0

    for pred, ref in zip(predictions, references):
        pred_set = set(pred.split())
        ref_set = set(ref.split())

        pred_set = {token.lower() for token in pred_set}

        pred_set = {token for token in pred_set if token in['a','b','c','d','e','f','g','h'
                                                            ,'i','j','k','l','m','n','o','p','q',
                                                            'r','s','t','u','v','w','x','y','z']}

        # True Positives, False Positives, False Negatives
        tp += len(pred_set & ref_set) 
        fp += len(pred_set - ref_set) 
        fn += len(ref_set - pred_set) 

        if pred_set == ref_set:
            exact_match_count += 1


    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    exact_match_accuracy = exact_match_count / len(references) if len(references) > 0 else 0.0

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match_accuracy": exact_match_accuracy,
    }