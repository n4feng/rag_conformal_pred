import numpy as np
import json
import csv
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
from math import ceil
from abc import ABC, abstractmethod

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
        for alpha in alphas:
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
        for alpha in alphas:
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

class ConditionalConformalCalibration(ConformalCalibration):
    """
    Implementation of conditional conformal calibration.
    """
    def calibrate_removal(self, dataset_prefixs, confidence_method, datasets, alphas, a, fig_filename, csv_filename):
        if confidence_method not in METHOD_SUPPORT_CONDITION:
            return
        
        print(f"Producing conditional conformal plot: {fig_filename}")
        # this will run almost same logic in ConformalCalibration, but with different calculate_correctness_and_removal implmentation in this class
        self._calibrate_removal(dataset_prefixs, confidence_method, datasets, alphas, a, fig_filename, csv_filename, suffix="_conditional")
    
    def calculate_correctness_and_removal(self, data, alphas, a, confidence_method):
        pre_defined_group = self._group_data(data)
        return self._compute_results(data, alphas, a, confidence_method, pre_defined_group)

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
        for alpha in tqdm(alphas):
            results_for_alpha = [[], []]
            for _ in range(1000):
                group_threshold_cache = {}
                random.shuffle(data)
                calibration_data, test_data = self.split_each_group(data)
                pre_defined_group = self._regroup_calibration_data(calibration_data)
                accepted_subclaim_list = self._get_group_accepted_subclaims(test_data, pre_defined_group, alpha, a, confidence_method, group_threshold_cache)
                fraction_correct = self._compute_fraction_correct(accepted_subclaim_list, a)
                results_for_alpha[0].append(1 - alpha)
                results_for_alpha[1].append(fraction_correct)
            results.append(results_for_alpha)
        return results

    # Make sure the split calibrate_range ratio are all same not just in overall level but in group level
    def split_each_group(self, data, calibrate_range=0.5):
        group_data = {}
        calibration_data = []
        test_data = []
        for entry in data:
            group = entry["groups"][0]  # Use first group as default
            group_data.setdefault(group, []).append(entry)
        for group_entries in group_data.values():
            split_index = ceil(len(group_entries) * calibrate_range)
            calibration_data.extend(group_entries[:split_index])
            test_data.extend(group_entries[split_index:])
        return calibration_data, test_data

    def _regroup_calibration_data(self, calibration_data):
        pre_defined_group = {}
        for item in calibration_data:
            for group in item['groups']:
                pre_defined_group.setdefault(group, []).append(item)
        return pre_defined_group

    # Calibrate based on group own threshold
    def _get_group_accepted_subclaims(self, test_data, pre_defined_group, alpha, a, confidence_method, group_threshold_cache):
        accepted_subclaim_list = []
        for test_data_point in test_data:
            threshold = 1.0
            for group in test_data_point['groups']:
                if group in group_threshold_cache:
                    group_tresh = group_threshold_cache[group]
                else:
                    group_tresh = compute_threshold(alpha, pre_defined_group[group], a, confidence_method)
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
            results = self.calculate_correctness_and_removal(data, alphas, a, confidence_method)
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


    def calculate_correctness_and_removal(self, data, alphas, a, confidence_method):
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
                pre_defined_group = self._regroup_calibration_data(calibration_data)
                accepted_subclaim_list, accepted_subclaim_list_pergroup = self._get_group_accepted_subclaims_with_groups(
                    test_data, pre_defined_group, alpha, a, confidence_method, group_threshold_cache
                )
                fraction_correct = self._compute_fraction_correct(accepted_subclaim_list, a)
                results_for_alpha[0].append(1 - alpha)
                results_for_alpha[1].append(fraction_correct)
                for group, accepted_list in accepted_subclaim_list_pergroup.items():
                    fraction = self._compute_fraction_correct(accepted_list, a)
                    results_for_alpha[2].setdefault(group, []).append(fraction)
            results.append(results_for_alpha)
        return results

    def _get_group_accepted_subclaims_with_groups(self, test_data, pre_defined_group, alpha, a, confidence_method, group_threshold_cache):
        accepted_subclaim_list = []
        accepted_subclaim_list_pergroup = {}
        for test_data_point in test_data:
            threshold = 1.0
            for group in test_data_point['groups']:
                if group in group_threshold_cache:
                    group_tresh = group_threshold_cache[group]
                else:
                    group_tresh = compute_threshold(alpha, pre_defined_group[group], a, confidence_method)
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