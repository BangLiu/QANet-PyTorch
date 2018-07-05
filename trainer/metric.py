"""
Official evaluation script for v1.1 of the SQuAD dataset.
Also added other defined metric functions.
"""
import json
import re
import string
import sys
import torch
import numpy as np
from collections import Counter


def normalize_answer(s):
    """
    Lower text and remove punctuation, articles and extra whitespace.
    :param s: original string
    :return: normalized string
    """
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    """
    Calculate F1 score given prediction and true answer strings.
    :param prediction: prediction string
    :param ground_truth: answer string
    :return: F1 score
    """
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    """
    Calculate exact match score given prediction and true answer strings.
    :param prediction: prediction string
    :param ground_truth: answer string
    :return: EM score
    """
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    """
    Calculate the maximum metric value when we have multiple ground truths.
    i.e., for each question, we have multiple answers.
    :param metric_fn: the function to calculate metric
    :param prediction: our model predicted answer string
    :param ground_truths: the list of answer strings
    :return: the maximum metric value by comparing our prediction
             to each ground_truth
    """
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(dataset, predictions):
    """
    Evaluate performance, calculate metrics EM and F1.
    :param dataset: the dictionary of 'data' in json file.
    :param predictions: the dictionary of our predictions.
                        (k, v) is like (qa['id'], prediction string)
    """
    f1 = exact_match = total = 0
    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                total += 1
                if qa['id'] not in predictions:
                    message = 'Unanswered question ' + qa['id'] + \
                              ' will receive score 0.'
                    print(message, file=sys.stderr)
                    continue
                ground_truths = list(map(lambda x: x['text'], qa['answers']))
                prediction = predictions[qa['id']]
                exact_match += metric_max_over_ground_truths(
                    exact_match_score, prediction, ground_truths)
                f1 += metric_max_over_ground_truths(
                    f1_score, prediction, ground_truths)

    exact_match = 100.0 * exact_match / total
    f1 = 100.0 * f1 / total

    return exact_match, f1


def evaluate_from_file(dataset_file, prediction_file):
    """
    Load dataset and prediction from two files, and evaluate
    the performance.
    """
    expected_version = '1.1'
    with open(dataset_file) as dataset_file:
        dataset_json = json.load(dataset_file)
        if dataset_json['version'] != expected_version:
            print('Evaluation expects v-' + expected_version +
                  ', but got dataset with v-' + dataset_json['version'],
                  file=sys.stderr)
        dataset = dataset_json['data']
    with open(prediction_file) as prediction_file:
        predictions = json.load(prediction_file)
    return evaluate(dataset, predictions)


def em_by_begin_end_index(pred_begin, pred_end, begin, end):
    """
    Calculate exact match score given the token index tensors of
    prediction boundary and true answer boundary.
    """
    batch_num = pred_begin.size(0)
    exact_correct_num = torch.sum(
        (pred_begin == begin) * (pred_end == end))
    em = exact_correct_num.item() / batch_num
    return em


def f1_by_begin_end_index(pred_begin, pred_end, begin, end):
    """
    Calculate F1 score given the token index tensors of
    prediction boundary and true answer boundary.
    """
    batch_size = pred_begin.size(0)
    f1_all = []
    for i in range(batch_size):
        pred = range(int(pred_begin[i]), int(pred_end[i] + 1))
        truth = range(int(begin[i]), int(end[i] + 1))
        overlap_len = len(list(set(pred) & set(truth)))
        pred_len = pred_end[i] - pred_begin[i] + 1
        truth_len = end[i] - begin[i] + 1

        precision = overlap_len / pred_len
        recall = overlap_len / truth_len
        if overlap_len == 0:
            f1 = 0
        else:
            f1 = ((2 * precision * recall) / (precision + recall)).item()
        f1_all.append(f1)
    f1 = np.mean(f1_all)
    return f1
