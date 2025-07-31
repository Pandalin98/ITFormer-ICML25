#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluation metrics for time series question answering.
Provides BLEU, ROUGE, accuracy, and RUL (Remaining Useful Life) metrics.
"""
from typing import List, Dict
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
from sklearn.metrics import accuracy_score, f1_score
from difflib import SequenceMatcher

def compute_bleu_from_ids(predictions, references):
    """Calculate BLEU score using strings.
    
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
    """Calculate ROUGE score using text.
    
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


def open_question_metrics(predictions, references, special_ids=[151643]):
    """Calculate BLEU and ROUGE scores for open-ended questions.
    
    Args:
        predictions (List[str]): Model predicted text.
        references (List[str]): Reference answer text.
        special_ids (List[int]): Special token IDs to filter out.

    Returns:
        Dict[str, float]: Scores containing BLEU and ROUGE.
    """
    decoded_predictions = []
    decoded_labels = []

    for pred, label in zip(predictions, references):
        pred = [token for token in pred if token not in special_ids]
        label = [token for token in label if token not in special_ids]
        decoded_predictions.append(pred)
        decoded_labels.append(label)

    bleu_score = compute_bleu_from_ids(predictions, references)
    rouge_scores = compute_rouge_from_ids(predictions, references)

    return {"BLEU": bleu_score, **rouge_scores}


def compute_rul(predictions, references):
    """Calculate RUL (Remaining Useful Life) score.

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

    return {"MAE": mae, "RMSE": rmse, "MSE": mse}


def closed_question_metrics(predictions, references, special_id=[151643]):
    """Calculate accuracy and F1 score for closed-ended questions.
    
    Args:
        predictions (List[str]): Model predicted text.
        references (List[str]): Reference answer text.
        special_id (List[int]): Special token IDs to filter out.

    Returns:
        Dict[str, float]: Scores containing accuracy and F1.
    """
    decoded_predictions = []
    decoded_labels = []

    for pred, label in zip(predictions, references):
        pred = [token for token in pred if token not in special_id]
        label = [token for token in label if token not in special_id]
        decoded_predictions.append(pred)
        decoded_labels.append(label)

    # Convert to strings for comparison
    pred_strings = [' '.join(map(str, pred)) for pred in decoded_predictions]
    ref_strings = [' '.join(map(str, label)) for label in decoded_labels]

    # Calculate exact match accuracy
    correct = sum(1 for pred, ref in zip(pred_strings, ref_strings) if pred.strip() == ref.strip())
    accuracy = correct / len(pred_strings) if pred_strings else 0.0

    # Calculate F1 score using sequence similarity
    f1_scores = []
    for pred, ref in zip(pred_strings, ref_strings):
        similarity = SequenceMatcher(None, pred.strip(), ref.strip()).ratio()
        f1_scores.append(similarity)
    
    avg_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0

    return {"Accuracy": accuracy, "F1": avg_f1}