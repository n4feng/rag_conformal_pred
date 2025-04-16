import numpy as np
import json
import csv
import random
import matplotlib.pyplot as plt
import os
import diskcache as dc
import math

from tqdm import tqdm
from math import ceil
from openai import OpenAI
from dotenv import load_dotenv
from calibration import ConformalCalibration
from calibration import calculate_calibration_distance, make_key, get_r_score, append_result_to_csv


def exponential_kernel(distance, h=1.0):
    return np.exp(-distance / (2 * h**2))

def conditional_coverage_group(data_path, m=3, n=3):
    """
    Divide data prompt into m * n bins where
    m is the number of groups true answers scores divided into
    n is the number of groups top subclaims scores divided into
    
    Returns:
        group: m x n grid of grouped queries
        true_answer_edges: list of m+1 bin edges for true_answer_score
        subclaim_edges: list of n+1 bin edges for top_subclaim_score
    """
    # Read the JSONL file into a list
    data = []
    with open(data_path, 'r') as file:
        for line in file:
            data.append(json.loads(line.strip()))

    true_answer_scores = [max(map(float, item['calibrate_score'])) for item in data]
    top_subclaim_scores = [max([score[1] for score in item["subclaims_score"]]) for item in data]

    min_true, max_true = min(true_answer_scores), max(true_answer_scores)
    min_sub, max_sub = min(top_subclaim_scores), max(top_subclaim_scores)

    interval_true = (max_true - min_true) / m
    interval_sub = (max_sub - min_sub) / n

    # Full bin edges, including min and max
    true_answer_edges = [min_true + i * interval_true for i in range(m + 1)]
    subclaim_edges = [min_sub + i * interval_sub for i in range(n + 1)]

    group = [[[] for _ in range(n)] for _ in range(m)]

    for i in range(len(data)):
        true_score = true_answer_scores[i]
        sub_score = top_subclaim_scores[i]

        # Bin index for true_score
        true_idx = m - 1  # default to last bin
        for k in range(m - 1):
            if true_score < true_answer_edges[k + 1]:
                true_idx = k
                break

        # Bin index for sub_score
        sub_idx = n - 1
        for k in range(n - 1):
            if sub_score < subclaim_edges[k + 1]:
                sub_idx = k
                break

        group[true_idx][sub_idx].append(data[i]["query"])

    return group, true_answer_edges, subclaim_edges

class LocalDatadependentCalibration(ConformalCalibration):
    """
    Local Calibration class for calibrating data-dependent models.
    """

    def __init__(self,embedding_model = "text-embedding-3-large"):
        """
        Initialize the LocalDatadependentCalibration class.

        Parameters:
        - model: The model to be calibrated.
        """
        self.embedding_model = embedding_model
        self.weights = {}

        self.prompt_embedding_cache = dc.Cache("data/cache/prompt_embedding")
        self.distance_cache = dc.Cache("data/cache/distance")
        dotenv_path = os.path.join(os.getcwd(), '.env')
        load_dotenv(dotenv_path)
        self.client = OpenAI()

    def build_weights(self, data):
        for i in tqdm(range(len(data))):
            total_weight = 0
            self.weights[i] = {}
            for j in range(len(data)):
                weight_key = data[i]["prompt"]+data[j]["prompt"]
                if i == j:
                    self.weights[weight_key] = 0
                else:
                    dis_key = make_key(data[i]["prompt"], data[j]["prompt"])
                    weight = 0
                    if dis_key in self.distance_cache:
                        weight = self.distance_cache[dis_key]
                    else:
                        weight = exponential_kernel(
                            calculate_calibration_distance(data[i], data[j], self.prompt_embedding_cache, self.client, self.embedding_model))
                        self.distance_cache[dis_key] = weight
                    self.weights[i][j] = weight
                    total_weight += weight
            #normalize weights
            for j in self.weights[i]:
                self.weights[i][j] /= total_weight

    def get_local_r_score(self, data, confidence_method, a):

        for i in range(len(data)):
            score = 0
            for j in range(len(data)):
                if i == j:
                    continue
                if get_r_score(data[i], confidence_method, a) > get_r_score(data[j], confidence_method, a):
                    score += self.weights[i][j]
            data[i]["local_r_score"] = score
        
    
    # Override common base functions used everywhere
    def _compute_threshold(self, alpha, calibration_data, test_data, a, confidence_method, pre_defined_group=False):
        """
        Computes the quantile/threshold from conformal prediction.
        # alpha: float in (0, 1)
        # r_scores: list of r_scores
        """
        # Compute threshold for conformal prection. The quantile is ceil((n+1)*(1-alpha))/n, and
        # We map this to the index by dropping the division by n and subtracting one (for zero-index).
        quantile_target_index = ceil((len(calibration_data) + 1) * (1 - alpha))
        local_threshold_datapoint = sorted(calibration_data, key=lambda x: x["local_r_score"])[quantile_target_index - 1]
        return get_r_score(local_threshold_datapoint, confidence_method, a)
            
    def _compute_results(self, data, alphas, a, confidence_method, pre_defined_group=None):
        self.build_weights(data)
        self.get_local_r_score(data, confidence_method,a)
        return super()._compute_results(data, alphas, a, confidence_method, pre_defined_group)
    
    def _process_calibration(self, data, alphas, a, confidence_method):
        self.build_weights(data)
        self.get_local_r_score(data, confidence_method,a)
        return super()._process_calibration(data, alphas, a, confidence_method)
    