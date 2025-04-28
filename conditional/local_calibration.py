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
from calibration import WeightedConformalCalibration
from calibration import calculate_calibration_distance, make_key, get_r_score, exponential_kernel

class LocalDatadependentCalibration(WeightedConformalCalibration):
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
        self.test_data_thresholds = {}
        
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
                    if dis_key in self.distance_cache:
                        distance = self.distance_cache[dis_key]
                    else:
                        distance = calculate_calibration_distance(data[i], data[j], self.prompt_embedding_cache, self.client, self.embedding_model)
                        self.distance_cache[dis_key] = distance
                    weight = exponential_kernel(distance,0.5)
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
    
    # def _process_partial_local(self, data, coverage_group, alpha, a, confidence_method, dataset_prefix, group_names=None):
    #     self.build_weights(data)
    #     self.get_local_r_score(data, confidence_method,a)
    #     return super()._process_partial_local(data, coverage_group, alpha, a, confidence_method, dataset_prefix, group_names)

    def _process_partial_local(self, data, coverage_group, alpha, a, confidence_method, dataset_prefix):
        self.build_weights(data)
        self.get_local_r_score(data, confidence_method,a)
        return super()._process_partial_local(data, coverage_group, alpha, a, confidence_method, dataset_prefix)