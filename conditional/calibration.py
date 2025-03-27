import numpy as np
import json
import csv
import random
import matplotlib.pyplot as plt
import os
import diskcache as dc
import math
from dotenv import load_dotenv
from tqdm import tqdm
from math import ceil
from abc import ABC, abstractmethod
from collections import defaultdict
from src.common.faiss_manager import FAISSIndexManager
from sklearn.metrics.pairwise import euclidean_distances
from openai import OpenAI


# Constants
METHOD_SUPPORT_CONDITION = ['similarity', 'gpt']
CORRECT_ANNOTATIONS = ["Y", "S"]

# Common base functions used everywhere
def compute_threshold(alpha, calibration_data, a, confidence_method):
    """
    Computes the quantile/threshold from conformal prediction.
    # alpha: float in (0, 1)
    # calibration_data: calibration data
    # a: as in paper, required fraction correct, section 4.1
    # confidence_method: string
    """
    # Compute r score for each example.
    r_scores = [get_r_score(entry, confidence_method, a) for entry in calibration_data]

    # Compute threshold for conformal prection. The quantile is ceil((n+1)*(1-alpha))/n, and
    # We map this to the index by dropping the division by n and subtracting one (for zero-index).
    quantile_target_index = ceil((len(r_scores) + 1) * (1 - alpha))
    threshold = sorted(r_scores)[quantile_target_index - 1]
    return threshold

# Common base functions used everywhere
def compute_weighted_threshold(alpha, calibration_data, calibration_data_weight, a, confidence_method):
    """
    Computes the quantile/threshold from conformal prediction.
    # alpha: float in (0, 1)
    # calibration_data: calibration data
    # calibration_data_weight: calibration data weight, same length as calibration_data
    # a: as in paper, required fraction correct, section 4.1
    # confidence_method: string
    """
    # Compute r score for each example.
    r_scores = [get_r_score(entry, confidence_method, a) for entry in calibration_data]

    return weighted_quantile(r_scores, calibration_data_weight, alpha)

def weighted_quantile(x, w, p):
    """
    Compute the accurate weighted (1 - p)th quantile by explicitly tilting the dataset.

    Parameters:
        x (array-like): intrinsic score value.
        w (array-like): Corresponding weights.
        p (float): Quantile level (e.g., 0.95 for 95th percentile).

    Returns:
        int or float: The exact weighted quantile.
    """
    # A hyper parameter temperature to control entropy of the weights
    TEMP = 5
    w = [math.exp(weight / TEMP) for weight in w]
    sum_w = sum(w)
    normalized_w = [weight / sum_w for weight in w] # Standard Softmax
    x, w = np.asarray(x), np.asarray(w)
    assert len(x) == len(normalized_w), "x and w must have the same length"
    assert 0 <= p <= 1, "p must be in [0, 1]"
    
    # Scale weights to determine repetition count
    len_w = len(normalized_w) #len_w/100 make sure each point in average been repeated 100 times
    repetitions = np.round([1/d/(len_w/100) for d in normalized_w]).astype(int)
    expanded_x = np.repeat(x, repetitions)
    expanded_x.sort()

    # Compute the (1 - p) quantile index
    quantile_index = int(np.floor((1 - p) * len(expanded_x)))
    return expanded_x[quantile_index]  # Return the exact quantile value

def get_r_score(entry, confidence_method, a):
    """
    Compute the r_a score for entry when confidence_method is used as the sub-claim scoring function.
    """
    #add a cache in entry to remember it's r_score during calibration
    r_score_key = f"r_score_{a}"
    if r_score_key in entry and entry[r_score_key]:
        return entry[r_score_key]

    threshold_set = sorted(
        [
            subclaim[confidence_method + "-score"] + subclaim.get("noise", 0)
            for subclaim in entry["claims"]
        ],
        reverse=True,
    )
    for threshold in threshold_set:
        curr_threshold = threshold
        # Apply threshold.
        accepted_subclaims = [
            subclaim
            for subclaim in entry["claims"]
            if subclaim[confidence_method + "-score"] + subclaim.get("noise", 0) >= threshold
        ]

        # Compute entailed/correct fraction.
        entailed_fraction = (
            np.mean(
                [
                    subclaim["annotation"] in CORRECT_ANNOTATIONS
                    for subclaim in accepted_subclaims
                ]
            )
            if accepted_subclaims
            else 1
        )

        if entailed_fraction < a:
            entry[r_score_key] = curr_threshold
            return curr_threshold
    entry[r_score_key] = -1
    return -1  # -1 is less than any score assigned by any of the implemented confidence methods

def load_calibration(filename="claims.jsonl"):
    """
    Reverse of dump_claims.
    """
    with open(filename, "r") as fopen:
        return json.load(fopen)["data"]

def dump_claims(output_list, filename="claims.jsonl"):
    """
    Dumps output_list into filename.
    [{"prompt": "Who is Tatsu?", "claims": [{"subclaim": "Tatsu is Japanese person", 'correct': 1.0}, {"subclaim": "Tatsu was born in 1988", 'correct': 0.0} ..]}]
    """
    with open(filename, "w") as outfile:
        merged_json = {"data": output_list}
        json.dump(merged_json, outfile, indent=4)


def append_result_to_csv(csv_filename, label, y, yerr):
    csvresult = [f"{y:.3f} ± {yerr:.3f}" for y, yerr in zip(y, yerr)]
    csvresult.reverse()
    row = [label] + csvresult
    with open(csv_filename, mode="a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(row)

def make_key(s1, s2):
    return tuple(sorted([s1, s2]))

def get_embedding(prompt, cache, client, model):
    """Retrieve the embedding for a given prompt, using a cache if available."""
    if prompt in cache:
        return cache[prompt]
    
    embedding = client.embeddings.create(input=[prompt], model=model)
    cache[prompt] = embedding
    return embedding

def calculate_calibration_distance(test_data, entry, cache, client, model):
    """Compute the Euclidean distance between test data and calibration data embeddings."""
    test_embedding = get_embedding(test_data["prompt"], cache, client, model)
    calibration_embedding = get_embedding(entry["prompt"], cache, client, model)

    test_vector = np.array(test_embedding.data[0].embedding).astype('float32').reshape(1, -1)
    calibration_vector = np.array(calibration_embedding.data[0].embedding).astype('float32').reshape(1, -1)

    return euclidean_distances(test_vector, calibration_vector)[0][0]

class ICalibration(ABC):
    """
    Interface for calibration methods.
    """
    @abstractmethod
    def calibrate_removal(self, dataset_prefixs, confidence_method, datasets, alphas, a, fig_filename, csv_filename):
        pass

    @abstractmethod
    def calibrate_factual(self, dataset_prefix, confidence_method, data, alphas, a, fig_filename, csv_filename):
        pass

    @abstractmethod
    def calibrate_partial_factual(self, dataset_prefix, confidence_method, data, alpha, a, fig_filename, csv_filename, group_size = 200):
        pass

class ConformalCalibration(ICalibration):
    """
    Implementation of standard conformal calibration.
    """
    def calibrate_removal(self, dataset_prefixs, confidence_method, datasets, alphas, a, fig_filename, csv_filename):
        print(f"Producing conformal plot: {fig_filename}")
        self._calibrate_removal(dataset_prefixs, confidence_method, datasets, alphas, a, fig_filename, csv_filename)

    def _calibrate_removal(self, dataset_prefixs, confidence_method, datasets, alphas, a, fig_filename, csv_filename, suffix=""):
        self._initialize_plot()
        self._write_csv_header(csv_filename, alphas)
        
        for dataset_prefix in dataset_prefixs:
            data = datasets[dataset_prefix]
            results = self._compute_results(data, alphas, a, confidence_method)
            self._process_results(dataset_prefix, results, csv_filename, suffix)
        
        self.finalize_removal_plot(dataset_prefixs, a, fig_filename)
    
    #Plotting part
    def _initialize_plot(self):
        plt.figure(dpi=800)

    def _write_csv_header(self, csv_filename, alphas):
        target_factuality = [f"{(1-x):.2f}" for x in alphas][::-1]
        header = ["dataset"] + target_factuality
        with open(csv_filename, mode="w", newline="") as file:
            csv.writer(file).writerow(header)

    def _process_results(self, dataset_prefix, results, csv_filename, suffix):
        x = [np.mean(results_for_alpha[0]) for results_for_alpha in results]
        y = [np.mean(results_for_alpha[1]) for results_for_alpha in results]
        yerr = [np.std(results_for_alpha[1]) * 1.96 / np.sqrt(len(results_for_alpha[1])) for results_for_alpha in results]
        label = dataset_prefix + suffix
        append_result_to_csv(csv_filename, label, y, yerr)
        plt.errorbar(x, y, yerr=yerr, label=label, linewidth=2)
    #Plotting part end

    def _compute_results(self, data, alphas, a, confidence_method, pre_defined_group=None):
        results = []
        for alpha in tqdm(alphas):
            results_for_alpha = [[], []]
            for i in range(len(data)):
                calibration_data = data[:i] + data[i + 1 :]
                test_data = data[i]
                threshold = self._compute_group_threshold(alpha, calibration_data, test_data, a, confidence_method, pre_defined_group)
                correctness, fraction_removed = self._evaluate_test_data(test_data, threshold, a, confidence_method)
                results_for_alpha[0].append(correctness)
                results_for_alpha[1].append(fraction_removed)
            results.append(results_for_alpha)
        return results

    def _compute_group_threshold(self, alpha, calibration_data, test_data, a, confidence_method, pre_defined_group=False):
        if pre_defined_group:
            return min(
                compute_threshold(alpha, pre_defined_group[group], a, confidence_method)
                for group in test_data['groups']
            )
        return compute_threshold(alpha, calibration_data, a, confidence_method)

    def _evaluate_test_data(self, test_data, threshold, a, confidence_method):
        accepted_subclaims = [
            subclaim for subclaim in test_data["claims"]
            if subclaim[confidence_method + "-score"] + subclaim.get("noise", 0) >= threshold
        ]
        fraction_removed = (
            0 if len(test_data["claims"]) == 0 else 1 - len(accepted_subclaims) / len(test_data["claims"])
        )
        correctness = (
            np.mean([subclaim["annotation"] in CORRECT_ANNOTATIONS for subclaim in accepted_subclaims]) >= a
            if accepted_subclaims else 1
        )
        return correctness, fraction_removed

    def finalize_removal_plot(self, dataset_prefixs, a, fig_filename):
        plt.xlabel(f"Fraction achieving avg factuality >= {a}" if a != 1 else "Fraction of factual outputs")
        plt.ylabel("Average percent removed")
        plt.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), fontsize=10)
        plt.title(f"Conformal Plots for {dataset_prefixs} Datasets (a={a})", fontsize=20)
        plt.savefig(fig_filename, bbox_inches="tight")

    def calibrate_factual(self, dataset_prefix, confidence_method, data, alphas, a, fig_filename, csv_filename):
        self._calibrate_and_plot(dataset_prefix, confidence_method, data, alphas, a, fig_filename, csv_filename)

    def _calibrate_and_plot(self, dataset_prefix, confidence_method, data, alphas, a, fig_filename, csv_filename, group_results=False):
        print(f"Producing calibration plot: {fig_filename}")
        x_values = np.linspace(1 - alphas[-1] - 0.05, 1 - alphas[0] + 0.03, 100)
        plt.plot(x_values, x_values, "--", color="gray", linewidth=2, label="Thrm 3.1 bounds")

        results = self._process_calibration(data, alphas, a, confidence_method)
        x = [np.mean(results_for_alpha[0]) for results_for_alpha in results]
        y = [np.mean(results_for_alpha[1]) for results_for_alpha in results]
        yerr = [np.std(results_for_alpha[1]) for results_for_alpha in results]

        append_result_to_csv(csv_filename, dataset_prefix, y, yerr)
        plt.plot(x, y, label=dataset_prefix, linewidth=2)

        if group_results:
            self._plot_group_results(results, dataset_prefix, csv_filename, x)

        self.finalize_factual_plot(fig_filename)

    def _plot_group_results(self, results, dataset_prefix, csv_filename, x):
        y_groups = {}
        y_groups_err = {}

        for result in results:
            for group, y_result in result[2].items():
                y_group = np.mean(y_result)
                y_group_err = np.std(y_result)
                y_groups.setdefault(group, []).append(y_group)
                y_groups_err.setdefault(group, []).append(y_group_err)

        for group, y_group in y_groups.items():
            label = f"{dataset_prefix}_conditional_{group}"
            append_result_to_csv(csv_filename, label, y_group, y_groups_err[group])
            plt.plot(x, y_group, label=label, linewidth=2)

    def finalize_factual_plot(self, fig_filename):
        plt.xlabel(f"Target factuality (1 - {chr(945)})", fontsize=16)
        plt.ylabel("Empirical factuality", fontsize=16)
        plt.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), fontsize=10)
        plt.savefig(fig_filename, bbox_inches="tight", dpi=800)
    
    def _process_calibration(self, data, alphas, a, confidence_method):
        results = []
        for alpha in tqdm(alphas):
            results_for_alpha = [[], []]
            for _ in range(1000):
                random.shuffle(data)
                split_index = len(data) // 2
                calibration_data = data[:split_index]
                test_data = data[split_index:]
                threshold = compute_threshold(alpha, calibration_data, a, confidence_method)
                accepted_subclaim_list = self._get_accepted_subclaims(test_data, threshold, confidence_method)
                fraction_correct = self._compute_fraction_correct(accepted_subclaim_list, a)
                results_for_alpha[0].append(1 - alpha)
                results_for_alpha[1].append(fraction_correct)
            results.append(results_for_alpha)
        return results

    def _get_accepted_subclaims(self, test_data, threshold, confidence_method):
        return [
            [subclaim for subclaim in test_data_point["claims"]
             if subclaim[confidence_method + "-score"] + subclaim.get("noise", 0) >= threshold]
            for test_data_point in test_data
        ]

    def _compute_fraction_correct(self, accepted_subclaim_list, a):
        entailed_fraction_list = [
            (
                np.mean(
                    [subclaim["annotation"] in CORRECT_ANNOTATIONS for subclaim in accepted_subclaims]
                ) if accepted_subclaims else 1
            )
            for accepted_subclaims in accepted_subclaim_list
        ]
        correctness_list = [entailed_fraction >= a for entailed_fraction in entailed_fraction_list]
        return sum(correctness_list) / len(correctness_list)
    
    def calibrate_partial_factual(self, dataset_prefix, confidence_method, data, alpha, a, fig_filename, csv_filename, group_size = 200):
        
        results = self._process_partial_calibration(data, alpha, a, confidence_method, group_size, dataset_prefix)
        overall_fraction = round(np.mean(results[-1]), 4)
        overall_std = round(np.std(results[-1]), 4)
        
        group_means = [round(np.mean(group), 4) for group in results[0:-1]]
        group_stds = [round(np.std(group), 4) for group in results[0:-1]]

        x_labels = ["Overall"] + [f"Group {i+1}" for i in range(len(results[0:-1]))]
        all_means = [overall_fraction] + group_means
        all_stds = [overall_std] + group_stds

        y_min = max(0, overall_fraction - 0.04)
        y_max = min(1, overall_fraction + 0.04)
            
        # Plotting
        plt.figure(figsize=(10, 6))
        plt.bar(x_labels, all_means, yerr=all_stds, color='blue', capsize=5)
        plt.axhline(y=1-alpha, color='gray', linestyle='--', linewidth=1.5, label=f'Alpha Level 1 - ({alpha:.2f})')
        plt.xlabel("Group")
        plt.ylabel("Fraction Correct")
        plt.title(f"Fraction Correct for Overall and Each Group (Group Size = {group_size})")
        plt.ylim(y_min, y_max)
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        
        # Save the plot to a file
        plt.savefig(fig_filename)
        plt.close()  # Close the plot to free up memory
        
        # Save the results to a CSV file
        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Group", "Mean Fraction Correct", "Std Fraction Correct"])
            writer.writerow(["Overall", overall_fraction, overall_std])
            for i, (mean, std) in enumerate(zip(group_means, group_stds)):
                writer.writerow([f"Group {i+1}", mean, std])

    def _process_partial_calibration(self, data, alpha, a, confidence_method, group_size, dataset_prefix):
        results = None
        for _ in range(1000):
            random.shuffle(data)
            split_index = len(data) // 2
            calibration_data = data[:split_index]
            test_data = data[split_index:]
            #group 0 to n-1 are partial test data, n is overall
            grouped_test_data = [test_data[i:i + group_size] for i in range(0, len(test_data), group_size)]
            if not results:
                results = [[] for _ in range(len(grouped_test_data)+1)]

            threshold = compute_threshold(alpha, calibration_data, a, confidence_method)
            for i, partial_test_data in enumerate(grouped_test_data):
                accepted_subclaim_list = self._get_accepted_subclaims(partial_test_data, threshold, confidence_method)
                fraction_correct = self._compute_fraction_correct(accepted_subclaim_list, a)
                results[i].append(fraction_correct)
                results[len(grouped_test_data)].append(fraction_correct)
        return results


class WeightedConformalCalibration(ConformalCalibration):
    def __init__(self, embedding_model = "text-embedding-3-large"):
        super().__init__()
        self.prompt_embedding_cache = dc.Cache("data/cache/prompt_embedding")
        dotenv_path = os.path.join(os.getcwd(), '.env')
        load_dotenv(dotenv_path)
        self.client = OpenAI()
        self.distance_cache = {} #key is  sorted than concat prompts
        self.embedding_model = embedding_model

    def _compute_results(self, data, alphas, a, confidence_method, pre_defined_group=None):
        results = []
        for alpha in tqdm(alphas):
            #record threshold per datapoint
            threshold_record = {}
            results_for_alpha = [[], []]
            for i in range(len(data)):
                calibration_data = data[:i] + data[i + 1 :]
                test_data = data[i]
                threshold = self._compute_group_threshold(alpha, calibration_data, test_data, a, confidence_method)
                threshold_record[test_data["prompt"]] = threshold
                correctness, fraction_removed = self._evaluate_test_data(test_data, threshold, a, confidence_method)
                results_for_alpha[0].append(correctness)
                results_for_alpha[1].append(fraction_removed)
            results.append(results_for_alpha)
            with open(f"data/out/weighted/threshold/threshold_record_{alpha:.2f}.json", "w") as fopen:
                json.dump(threshold_record, fopen, indent=4)
        return results
    
    def _compute_group_threshold(self, alpha, calibration_data, test_data, a, confidence_method, pre_defined_group=None):
        distance = []
        for entry in calibration_data:
            key = make_key(entry["prompt"], test_data["prompt"])
            if key in self.distance_cache:
                distance.append(self.distance_cache[key])
            else:
                cal_distance = calculate_calibration_distance(test_data, entry, self.prompt_embedding_cache, self.client, self.embedding_model)
                self.distance_cache[key] = cal_distance
                distance.append(cal_distance)
        return compute_weighted_threshold(alpha, calibration_data, distance, a, confidence_method)
    
    def _process_calibration(self, data, alphas, a, confidence_method):
        results = []
        for alpha in tqdm(alphas):
            results_for_alpha = [[], []]
            for _ in range(1000):
                random.shuffle(data)
                split_index = len(data) // 2
                pickup_calibration =1000
                if split_index <= pickup_calibration:
                    calibration_data = data[:split_index]
                else:
                    #randonly select n data points
                    calibration_data = random.sample(data[:split_index], pickup_calibration)
                test_data = data[split_index:]
                accepted_subclaim_list = []
                for entry in test_data:
                    threshold = self._compute_group_threshold(alpha, calibration_data, entry, a, confidence_method)
                    accepted_subclaim_list.append(
                        [subclaim for subclaim in entry["claims"]
                         if subclaim[confidence_method + "-score"] + subclaim.get("noise", 0) >= threshold]
                    )
                fraction_correct = self._compute_fraction_correct(accepted_subclaim_list, a)
                results_for_alpha[0].append(1 - alpha)
                results_for_alpha[1].append(fraction_correct)
            results.append(results_for_alpha)
        return results
    
    def _load_threshold(self, filename):
        with open(filename, "r") as fopen:
            return json.load(fopen)
        
    def _process_partial_calibration(self, data, alpha, a, confidence_method, group_size, dataset_prefix):
        results = None
        for _ in range(1000):
            random.shuffle(data)
            split_index = len(data) // 2
            pickup_calibration = 200
            if split_index <= pickup_calibration:
                calibration_data = data[:split_index]
            else:
                #randonly select 100 data points
                calibration_data = random.sample(data[:split_index], pickup_calibration)
            test_data = data[split_index:]
            grouped_test_data = [test_data[i:i + group_size] for i in range(0, len(test_data), group_size)]
            if not results:
                #group 0 to n-1 are partial test data, n is overall
                results = [[] for _ in range(len(grouped_test_data)+1)]

            for i, partial_test_data in enumerate(grouped_test_data):
                accepted_subclaim_list = []
                for entry in partial_test_data:
                    threshold = self._compute_group_threshold(alpha, calibration_data, entry, a, confidence_method)
                    accepted_subclaim_list.append([subclaim for subclaim in entry["claims"]
                         if subclaim[confidence_method + "-score"] + subclaim.get("noise", 0) >= threshold]
                    )
                fraction_correct = self._compute_fraction_correct(accepted_subclaim_list, a)
                results[i].append(fraction_correct)
                results[len(grouped_test_data)].append(fraction_correct)
        return results

class DistanceConformalCalibration(ConformalCalibration):
    def __init__(self, embedding_model = "text-embedding-3-large"):
        super().__init__()
        self.prompt_embedding_cache = dc.Cache("data/cache/prompt_embedding")
        dotenv_path = os.path.join(os.getcwd(), '.env')
        load_dotenv(dotenv_path)
        self.client = OpenAI()
        self.distance_cache = {} #key is  sorted than concat prompts
        self.embedding_model = embedding_model
    
    def form_calibration_set(self, test_entry, calibration_data, calibration_size=100):
        if len(calibration_data) < calibration_size:
            return calibration_data
        calibration_data_with_distance = []
        for entry in calibration_data:
            key = make_key(entry["prompt"], test_entry["prompt"])
            if key in self.distance_cache:
                distance = self.distance_cache[key]
            else:
                distance = calculate_calibration_distance(test_entry, entry, self.prompt_embedding_cache, self.client, self.embedding_model)
                self.distance_cache[key] = distance
            calibration_data_with_distance.append((entry, distance))

        calibration_data_with_distance.sort(key=lambda x: x[1])
        return [entry for entry, _ in calibration_data_with_distance[:calibration_size]]
    
    def _compute_group_threshold(self, alpha, calibration_data, test_data, a, confidence_method, pre_defined_group=None):
        calibration_data = self.form_calibration_set(test_data, calibration_data)
        return compute_threshold(alpha, calibration_data, a, confidence_method)
    
    def _process_calibration(self, data, alphas, a, confidence_method):
        results = []
        for alpha in tqdm(alphas):
            results_for_alpha = [[], []]
            for _ in range(1000):
                random.shuffle(data)
                split_index = len(data) // 2
                calibration_data = data[:split_index]
                test_data = data[split_index:]
                accepted_subclaim_list = []
                for entry in test_data:
                    threshold = self._compute_group_threshold(alpha, calibration_data, entry, a, confidence_method)
                    accepted_subclaim_list.append(
                        [subclaim for subclaim in entry["claims"]
                         if subclaim[confidence_method + "-score"] + subclaim.get("noise", 0) >= threshold]
                    )
                fraction_correct = self._compute_fraction_correct(accepted_subclaim_list, a)
                results_for_alpha[0].append(1 - alpha)
                results_for_alpha[1].append(fraction_correct)
            results.append(results_for_alpha)
        return results

class ConditionalConformalCalibration(ConformalCalibration):
    """
    Implementation of conditional conformal calibration.
    """
    def calibrate_removal(self, dataset_prefixs, confidence_method, datasets, alphas, a, fig_filename, csv_filename):
        if confidence_method not in METHOD_SUPPORT_CONDITION:
            return
        
        print(f"Producing conditional conformal plot: {fig_filename}")
        self._calibrate_removal(dataset_prefixs, confidence_method, datasets, alphas, a, fig_filename, csv_filename, suffix="_conditional")

    def _compute_results(self, data, alphas, a, confidence_method, pre_defined_group=None):
        pre_defined_group = self._group_data(data)
        return super()._compute_results(data, alphas, a, confidence_method, pre_defined_group)

    def _group_data(self, data):
        pre_defined_group = {}
        for item in data:
            for group in item['groups']:
                pre_defined_group.setdefault(group, []).append(item)
        return pre_defined_group
    
    def calibrate_factual(self, dataset_prefix, confidence_method, data, alphas, a, fig_filename, csv_filename):
        if confidence_method in METHOD_SUPPORT_CONDITION:
            self._calibrate_and_plot(dataset_prefix + "_conditional", confidence_method, data, alphas, a, fig_filename, csv_filename)
        else:
            raise ValueError("Conditional calibration only supports similarity and GPT confidence methods")
    
    def _process_calibration(self, data, alphas, a, confidence_method):
        results = []
        for alpha in alphas:
            results_for_alpha = [[], []]
            for _ in range(1000):
                group_threshold_cache = {}
                random.shuffle(data)
                calibration_data, test_data = self.split_each_group(data)
                accepted_subclaim_list = self._get_group_accepted_subclaims(test_data, calibration_data, alpha, a, confidence_method, group_threshold_cache)
                fraction_correct = self._compute_fraction_correct(accepted_subclaim_list, a)
                results_for_alpha[0].append(1 - alpha)
                results_for_alpha[1].append(fraction_correct)
            results.append(results_for_alpha)
        return results

    # Make sure the split calibrate_range ratio are all same not just in overall level but in group level
    # not return data in list but in a map with each group name as key
    def split_each_group(self, data, calibrate_range=0.5):
        group_data = defaultdict(list)
        calibration_data = defaultdict(list)
        test_data = []

        for entry in data:
            group = entry["groups"][0]  # Use first group as default
            group_data[group].append(entry)

        for group, group_entries in group_data.items():
            split_index = ceil(len(group_entries) * calibrate_range)
            calibration_data[group].extend(group_entries[:split_index])
            test_data.extend(group_entries[split_index:])

        return calibration_data, test_data

    # Calibrate based on group own threshold
    def _get_group_accepted_subclaims(self, test_data, calibration_data, alpha, a, confidence_method, group_threshold_cache):
        accepted_subclaim_list = []
        for test_data_point in test_data:
            threshold = 1.0
            for group in test_data_point['groups']:
                if group in group_threshold_cache:
                    group_tresh = group_threshold_cache[group]
                else:
                    group_tresh = compute_threshold(alpha, calibration_data[group], a, confidence_method)
                    group_threshold_cache[group] = group_tresh
                threshold = min(threshold, group_tresh)
            accepted_subclaim_list.append(
                [subclaim for subclaim in test_data_point["claims"]
                 if subclaim[confidence_method + "-score"] + subclaim.get("noise", 0) >= threshold]
            )
        return accepted_subclaim_list
    
class ConditionalConformalCalibrationWithGroup(ConditionalConformalCalibration):
    """
    Implementation of conditional conformal calibration with group.
    """
    def calibrate_removal(self, dataset_prefixs, confidence_method, datasets, alphas, a, fig_filename, csv_filename):
        print(f"Producing conformal plot: {fig_filename}")
        self._initialize_plot()
        self._write_csv_header(csv_filename, alphas)
        
        for dataset_prefix in tqdm(dataset_prefixs):
            if confidence_method not in METHOD_SUPPORT_CONDITION:
                continue
            data = datasets[dataset_prefix]
            results = self._compute_results(data, alphas, a, confidence_method)
            self._process_results(dataset_prefix, results, csv_filename, suffix="_conditional")
            self._process_group_results(dataset_prefix, results, csv_filename)
        
        self.finalize_removal_plot(dataset_prefixs, a, fig_filename)

    def _process_group_results(self, dataset_prefix, results, csv_filename):
        y_groups, y_groups_err = {}, {}
        for result in results:
            for group, y_result in result[2].items():
                y_group = np.mean(y_result)
                y_group_err = np.std(y_result) * 1.96 / np.sqrt(len(y_result))
                y_groups.setdefault(group, []).append(y_group)
                y_groups_err.setdefault(group, []).append(y_group_err)
        
        for group, y_group in y_groups.items():
            label = f"{dataset_prefix}_conditional_{group}"
            append_result_to_csv(csv_filename, label, y_group, y_groups_err[group])
            plt.errorbar([np.mean(result[0]) for result in results], y_group, yerr=y_groups_err[group], label=label, linewidth=2)

        self._plot_base_factuality_point(results)

    def _plot_base_factuality_point(self, results):
        x_point, y_point = np.mean(results[-1][0]), np.mean(results[-1][1])
        plt.scatter(x_point, y_point, color="black", marker="*", s=235, label="Base factuality", zorder=1000)


    def _compute_results(self, data, alphas, a, confidence_method):
        """
        Calculates correctness and fraction removed for a dataset over a range of alphas.

        Args:
            data (list): The dataset, where each entry contains group tags, claims and annotations.
            predifned_group (list): List of predefined groups for threshold computation.
            alphas (list): List of alpha values for threshold computation.
            a (float): Minimum entailed fraction threshold for correctness.
            confidence_method (str): The method used to compute confidence.

        Returns:
            list: Results containing average correctness and fraction removed for each alpha.
        """
        pre_defined_group = self._group_data(data)
        results = []
        for alpha in alphas:
            results_for_alpha = [[], [], {}]
            for i in range(len(data)):
                test_data = data[i]
                threshold = self._compute_group_threshold(alpha, None, test_data, a, confidence_method, pre_defined_group)
                correctness, fraction_removed = self._evaluate_test_data(test_data, threshold, a, confidence_method)
                results_for_alpha[0].append(correctness)
                results_for_alpha[1].append(fraction_removed)
                for group in test_data['groups']:
                    results_for_alpha[2].setdefault(group, []).append(fraction_removed)
            #print(f"Processing for alpha {alpha} done")
            results.append(results_for_alpha)
        return results
    
    def calibrate_factual(self, dataset_prefix, confidence_method, data, alphas, a, fig_filename, csv_filename):
        if confidence_method in METHOD_SUPPORT_CONDITION:
            self._calibrate_and_plot(dataset_prefix + "_conditional", confidence_method, data, alphas, a, fig_filename, csv_filename, group_results=True)
        else:
            raise ValueError("Conditional calibration only supports similarity and GPT confidence methods")


    def _plot_group_results(self, results, dataset_prefix, csv_filename, x):
        y_groups = {}
        y_groups_err = {}

        for result in results:
            for group, y_result in result[2].items():
                y_group = np.mean(y_result)
                y_group_err = np.std(y_result)
                y_groups.setdefault(group, []).append(y_group)
                y_groups_err.setdefault(group, []).append(y_group_err)

        for group, y_group in y_groups.items():
            label = f"{dataset_prefix}_conditional_{group}"
            append_result_to_csv(csv_filename, label, y_group, y_groups_err[group])
            plt.plot(x, y_group, label=label, linewidth=2)
    
    def _process_calibration(self, data, alphas, a, confidence_method):
        results = []
        for alpha in tqdm(alphas):
            results_for_alpha = [[], [], {}]
            for _ in range(1000):
                group_threshold_cache = {}
                random.shuffle(data)
                calibration_data, test_data = self.split_each_group(data)
                accepted_subclaim_list, accepted_subclaim_list_pergroup = self._get_group_accepted_subclaims_with_groups(
                    test_data, calibration_data, alpha, a, confidence_method, group_threshold_cache
                )
                fraction_correct = self._compute_fraction_correct(accepted_subclaim_list, a)
                results_for_alpha[0].append(1 - alpha)
                results_for_alpha[1].append(fraction_correct)
                for group, accepted_list in accepted_subclaim_list_pergroup.items():
                    fraction = self._compute_fraction_correct(accepted_list, a)
                    results_for_alpha[2].setdefault(group, []).append(fraction)
            results.append(results_for_alpha)
        return results

    def _get_group_accepted_subclaims_with_groups(self, test_data, calibration_data, alpha, a, confidence_method, group_threshold_cache):
        accepted_subclaim_list = []
        accepted_subclaim_list_pergroup = {}
        for test_data_point in test_data:
            threshold = 1.0
            for group in test_data_point['groups']:
                if group in group_threshold_cache:
                    group_tresh = group_threshold_cache[group]
                else:
                    group_tresh = compute_threshold(alpha, calibration_data[group], a, confidence_method)
                    group_threshold_cache[group] = group_tresh
                    accepted_subclaim_list_pergroup[group] = []
                threshold = min(threshold, group_tresh)
            accepted_subclaims = [
                subclaim for subclaim in test_data_point["claims"]
                if subclaim[confidence_method + "-score"] + subclaim.get("noise", 0) >= threshold
            ]
            accepted_subclaim_list.append(accepted_subclaims)
            for group in test_data_point['groups']:
                accepted_subclaim_list_pergroup[group].append(accepted_subclaims)
        return accepted_subclaim_list, accepted_subclaim_list_pergroup
    
class DynamicConditionalConformalCalibration(ConditionalConformalCalibration):
    """
    First dynamicly form group of calibration data based on retrived document type
    Then use that group to calculate threshold for test data
    """
    def __init__(self):
        self.faiss_manager = FAISSIndexManager()
        self.test_data_thresholds = {}
        self.doc_fraction_cache = {}

    def calibrate_removal(self, dataset_prefixs, confidence_method, datasets, alphas, a, fig_filename, csv_filename):
        if confidence_method not in METHOD_SUPPORT_CONDITION:
            return
        
        print(f"Producing conditional conformal plot: {fig_filename}")
        self._calibrate_removal(dataset_prefixs, confidence_method, datasets, alphas, a, fig_filename, csv_filename, suffix="_conditional")
        
    def _dynamic_group_data(self, entry, group_data, group_size=500):
        calibration_data = []
        group_target_size = {}
        if entry["prompt"] in self.doc_fraction_cache:
            group_target_size =  self.doc_fraction_cache[entry["prompt"]]
        else:
            retrieved_docs = self.faiss_manager.search_faiss_index(entry["prompt"], 10, 0.3)
            parsed_docs = [self.faiss_manager.parse_result(doc) for doc in retrieved_docs]
            #count each filepath in metadata
            doc_count = defaultdict(int)
            for doc in parsed_docs:  
                doc_count[doc['metadata']['file_path']] += 1
            total_docs = len(parsed_docs)
            #calculate fraction of each group
            group_target_size = {k: ceil(v / total_docs * group_size) for k, v in doc_count.items()}
            self.doc_fraction_cache[entry["prompt"]] = group_target_size

        # Randomly sample the data to match target size
        for filename, size in group_target_size.items():
            for group in group_data.keys():
                if group.lower() in filename.lower():
                    if len(group_data[group]) < size:
                        calibration_data.extend(group_data[group])
                    else:
                        calibration_data.extend(random.sample(group_data[group], size))
        
        return calibration_data
    
    def _compute_results(self, data, alphas, a, confidence_method, pre_defined_group=None):
        results = []
        group_data = defaultdict(list)
        for item in data:
            groups = item['groups']
            for group in groups:
                group_data[group].append(item)
        for alpha in tqdm(alphas):
            results_for_alpha = [[], []]
            for i in range(len(data)):
                test_data = data[i]
                # dynamicly form calibration data
                calibration_data = self._dynamic_group_data(test_data, group_data)
                threshold = compute_threshold(alpha, calibration_data, a, confidence_method)
                correctness, fraction_removed = self._evaluate_test_data(test_data, threshold, a, confidence_method)
                results_for_alpha[0].append(correctness)
                results_for_alpha[1].append(fraction_removed)
            results.append(results_for_alpha)
        return results
    
    def _process_calibration(self, data, alphas, a, confidence_method):
        results = []
        for alpha in tqdm(alphas):
            #too time consuming to do customize calibration for each test data
            results_for_alpha = [[], []]
            for _ in range(1000):
                random.shuffle(data)
                calibration_data, test_data = self.split_each_group(data)
                accepted_subclaim_list = self._get_group_accepted_subclaims(test_data, calibration_data, alpha, a, confidence_method)
                fraction_correct = self._compute_fraction_correct(accepted_subclaim_list, a)
                results_for_alpha[0].append(1 - alpha)
                results_for_alpha[1].append(fraction_correct)
            results.append(results_for_alpha)
            with open(f"data/out/dynamic/test_data_thresholds_alpha={alpha:.2f}.json", "w") as fopen:
                json.dump(self.test_data_thresholds, fopen)
            self.test_data_thresholds = {}
        return results   
    
    # Calibrate based on each test data customized calibration set
    def _get_group_accepted_subclaims(self, test_data, calibration_data, alpha, a, confidence_method):
        accepted_subclaim_list = []
        for test_data_point in test_data:
            selected_calibration_data = self._dynamic_group_data(test_data_point, calibration_data)
            threshold = compute_threshold(alpha, selected_calibration_data, a, confidence_method)
            # Track threshold per test_data["prompt"]
            prompt = test_data_point["prompt"]
            if prompt not in self.test_data_thresholds:
                self.test_data_thresholds[prompt] = []
            self.test_data_thresholds[prompt].append(f"{threshold:.5f}")
            accepted_subclaim_list.append(
                [subclaim for subclaim in test_data_point["claims"]
                 if subclaim[confidence_method + "-score"] + subclaim.get("noise", 0) >= threshold]
            )
        return accepted_subclaim_list
