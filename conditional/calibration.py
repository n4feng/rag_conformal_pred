import numpy as np
import json
import csv
import random
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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

def exponential_kernel(distance, h=1.0):
    return np.exp(-distance / (2 * h**2))

def weighted_quantile(x, d, p):
    """
    Compute the accurate weighted (1 - p)th quantile by explicitly tilting the dataset.

    Parameters:
        x (array-like): intrinsic score value.
        d (array-like): Corresponding distance.
        p (float): Quantile level (e.g., 0.95 for 95th percentile).

    Returns:
        int or float: The exact weighted quantile.
    """
    # Smoothing parameter for exponential kernel
    h = 0.7
    w = [exponential_kernel(distance, h) for distance in d]
    sum_w = sum(w)
    normalized_w = [weight / sum_w for weight in w] # Standard Softmax
    x, w = np.asarray(x), np.asarray(w)
    assert len(x) == len(normalized_w), "x and w must have the same length"
    assert 0 <= p <= 1, "p must be in [0, 1]"
    
    # Scale weights to determine repetition count
    len_w = len(normalized_w) #len_w/100 make sure each point in average been repeated 100 times
    repetitions = np.round([n_weight*len_w*100 for n_weight in normalized_w]).astype(int)
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

# def pooled_mean_std(group_sizes, group_means, group_stds):
#     """
#     Compute overall mean and standard deviation from group sizes, means, and stds.

#     Parameters:
#         group_sizes (list of int): Number of elements in each group
#         group_means (list of float): Mean of each group
#         group_stds (list of float): Standard deviation of each group

#     Returns:
#         tuple: (overall_mean, overall_std)
#     """
#     group_sizes = np.array(group_sizes)
#     group_means = np.array(group_means)
#     group_stds = np.array(group_stds)
    
#     N = np.sum(group_sizes)
#     if N == 0:
#         return float('nan'), float('nan')

#     # Weighted average for the overall mean
#     overall_mean = np.sum(group_sizes * group_means) / N

#     # Total variance includes within-group and between-group components
#     total_variance = np.sum(
#         group_sizes * (group_stds**2 + (group_means - overall_mean)**2)
#     ) / N

#     overall_std = np.sqrt(total_variance)

#     return overall_mean, overall_std

def get_embedding(input_string, cache, client, model):
    """Retrieve the embedding for a given prompt, using a cache if available."""
    if input_string in cache:
        return cache[input_string]
    
    embedding = client.embeddings.create(input=[input_string], model=model)
    cache[input_string] = embedding
    return embedding

def calculate_calibration_distance(test_data, entry, cache, client, model):
    """Compute the Euclidean distance between test data and calibration data embeddings."""
    test_embedding = get_embedding(test_data["prompt"] + test_data["original-output"], cache, client, model)
    calibration_embedding = get_embedding(entry["prompt"] + entry["original-output"], cache, client, model)

    test_vector = np.array(test_embedding.data[0].embedding).astype('float32').reshape(1, -1)
    calibration_vector = np.array(calibration_embedding.data[0].embedding).astype('float32').reshape(1, -1)
    euc_dis = euclidean_distances(test_vector, calibration_vector)[0][0]
    return euc_dis

# def divide_coverage_group(data_path, m=3, n=3):
#     """
#     Divide data prompt into m * n bins where
#     m is the number of groups top subclaims scores divided into (y, row idx)
#     n is the number of groups true answers scores divided into (x, column idx )
    
#     Returns:
#         group: m x n grid of grouped queries
#         true_answer_edges: list of m+1 bin edges for true_answer_score
#         subclaim_edges: list of n+1 bin edges for top_subclaim_score
#     """
#     # Read the JSONL file into a list
#     data = []
#     with open(data_path, 'r') as file:
#         for line in file:
#             data.append(json.loads(line.strip()))

#     true_answer_scores = [max(map(float, item['calibrate_score'])) for item in data]
#     top_subclaim_scores = [max([score[1] for score in item["subclaims_score"]]) for item in data]

#     min_true, max_true = min(true_answer_scores), max(true_answer_scores)
#     min_sub, max_sub = min(top_subclaim_scores), max(top_subclaim_scores)

#     interval_true = (max_true - min_true) / n
#     interval_sub = (max_sub - min_sub) / m

#     # Full bin edges, including min and max
#     subclaim_edges = [min_sub + i * interval_sub for i in range(m + 1)]
#     true_answer_edges = [min_true + i * interval_true for i in range(n + 1)]

#     group = [[[] for _ in range(n)] for _ in range(m)]

#     for i in range(len(data)):
#         true_score = true_answer_scores[i]
#         sub_score = top_subclaim_scores[i]

#         # Bin index for true_score
#         sub_idx = m - 1  # default to last bin
#         for k in range(m - 1):
#             if sub_score < subclaim_edges[k + 1]:
#                 sub_idx = k
#                 break

#         # Bin index for sub_score
#         true_idx = n - 1
#         for k in range(n - 1):
#             if true_score < true_answer_edges[k + 1]:
#                 true_idx = k
#                 break

#         group[sub_idx][true_idx].append(data[i]["query"])
    
#     # Reverse the order of the rows in the group 
#     group = group[::-1]
#     # Flip subclaim_edges list and use 1-origin as un-certainty
#     subclaim_edges = [1 - edge for edge in subclaim_edges][::-1]

#     return group, subclaim_edges, true_answer_edges

# def divide_by_group(data_path, group_names, m=5):
#     """
#     Divide data prompt into m * n bins where
#     group_names will be x axis
#     m is the number of groups top subclaims scores divided into (y, row idx)
    
    
#     Returns:
#         group: m x len(group_names) grid of grouped queries
#         subclaim_edges: list of n+1 bin edges for top_subclaim_score
#     """
#     # Read the JSONL file into a list
#     data = []
#     with open(data_path, 'r') as file:
#         for line in file:
#             data.append(json.loads(line.strip()))

#     top_subclaim_scores = [1 - max([score[1] for score in item["subclaims_score"]]) for item in data]

#     min_sub, max_sub = min(top_subclaim_scores), max(top_subclaim_scores)

#     interval_sub = (max_sub - min_sub) / m

#     # Full bin edges, including min and max
#     subclaim_edges = [min_sub + i * interval_sub for i in range(m + 1)]

#     group = [{name: [] for name in group_names} for _ in range(m)]

#     for i in range(len(data)):
#         sub_score = top_subclaim_scores[i]

#         # Bin index for true_score
#         sub_idx = m - 1  # default to last bin

#         for k in range(m - 1):
#             if sub_score < subclaim_edges[k + 1]:
#                 sub_idx = k
#                 break
#         group_name = data[i]["groups"][0]
#         group[sub_idx][group_name].append(data[i]["query"])
    
#     # Reverse the order of the rows in the group 
#     group = group[::-1]


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

    #@abstractmethod
    # def calibrate_factual_local(self, dataset_prefix, score_filepath, confidence_method, data, alpha, a, fig_filename, csv_filename, m, n):
    #     pass
    @abstractmethod
    def calibrate_factual_pergroup(self, dataset_prefix, confidence_method, data, alpha, a, fig_filename, csv_filename, catogories, group_names):
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

    # def plot_stat_grid(self, coverage_group, data_cells, x_edges, y_edges, alpha, fig_filename, dataset_prefix):
    #     """
    #     Plot a grid where each cell contains a list of values between 0-1 or None.
    #     Each cell displays the mean ± std (or N/A) and is colored by the mean.
        
    #     Parameters:
    #         coverage_group (2D list): m x n grid of queries
    #         data_cells (2D list): m x n grid, each element is a list of floats or None.
    #         x_edges (1D list or array): n+1 numerical edges or list of n group names.
    #         y_edges (1D np.ndarray): m+1 numerical edges in Y.
    #     """
    #     m, n = len(data_cells), len(data_cells[0])
        
    #     is_named_x = isinstance(x_edges[0], str)

    #     if is_named_x:
    #         # Create dummy edges for categorical x-axis labels
    #         x_positions = np.arange(n + 1)
    #         x_centers = (x_positions[:-1] + x_positions[1:]) / 2
    #         x_ticks = x_centers
    #         x_tick_labels = x_edges
    #         x_edges_plot = x_positions
    #     else:
    #         x_edges_plot = x_edges

    #     # Compute mean values and std for color and labels
    #     mean_grid = np.empty((m, n))
    #     labels = [["" for _ in range(n)] for _ in range(m)]
        
    #     for i in range(m):
    #         for j in range(n):
    #             data_j = x_edges[j] if is_named_x else j
    #             cell = data_cells[i][data_j]
    #             if cell is None or len(cell) == 0:
    #                 mean_grid[i, j] = 1 - alpha  # Color as white default
    #                 labels[i][j] = "N/A\nCount: 0"
    #             else:
    #                 arr = np.array(cell)
    #                 mean = arr.mean()
    #                 std = arr.std()
    #                 mean_grid[i, j] = mean
    #                 labels[i][j] = f"{mean:.2f} ± {std:.2f}\nCount: {len(coverage_group[i][data_j])}"

    #     # Colormap from bright red → white → mint green
    #     cmap = mcolors.LinearSegmentedColormap.from_list(
    #         "red_white_green",
    #         [
    #             (0.0, (1.0, 0.2, 0.2)),
    #             ((0.3 - alpha) / 0.3, (1.0, 1.0, 0.95)),
    #             (1.0, (0.6, 1.0, 0.7))
    #         ]
    #     )
    #     norm = mcolors.Normalize(vmin=0.7, vmax=1.0, clip=True)

    #     # Plot
    #     fig, ax = plt.subplots()
    #     mesh = ax.pcolormesh(x_edges_plot, y_edges, mean_grid, cmap=cmap, norm=norm,
    #                         edgecolors='k', linewidth=0.5, shading='auto')

    #     # Add text annotations within grid cells
    #     for i in range(m):
    #         for j in range(n):
    #             if is_named_x:
    #                 x = (x_edges_plot[j] + x_edges_plot[j+1]) / 2
    #             else:
    #                 x = (x_edges[j] + x_edges[j+1]) / 2
    #             y = (y_edges[i] + y_edges[i+1]) / 2
    #             ax.text(x, y, labels[i][j], ha='center', va='center', fontsize=8)

    #     # Axis labels and colorbar
    #     cbar = plt.colorbar(mesh, ax=ax)
    #     cbar.set_label("Mean Value (Color scale: Red → White → Green)")
        
    #     ax.set_xlabel('Predefined Groups' if is_named_x else 'True Answer Score Group')
    #     ax.set_ylabel('1 - max Subcalaim Score as Uncertainty')
    #     ax.set_title(dataset_prefix)
    #     ax.set_aspect('auto')
    #     ax.invert_yaxis()

    #     # Handle categorical x labels
    #     if is_named_x:
    #         ax.set_xticks(x_ticks)
    #         ax.set_xticklabels(x_tick_labels, rotation=45, ha='right')
    #     else:
    #         x_centers = [(x_edges[i] + x_edges[i+1]) / 2 for i in range(len(x_edges) - 1)]
    #         ax.set_xticks(x_centers)
    #         ax.set_xticklabels([f"{x_center:.2f}" for x_center in x_centers])
            

    #     plt.savefig(fig_filename, bbox_inches="tight", dpi=800)
    #     plt.show()


    def plot_stat_grid(self, coverage_group, results, group_names, categories, alpha, fig_filename, dataset_prefix):
        """
        Plot a grid where each cell contains a list of values between 0-1 or None.
        Each cell displays the mean ± std (or N/A) and is colored by the mean.

        Parameters:
            coverage_group (2D list): m x n grid of queries
            data_cells (2D list): m x n grid, each element is a list of floats or None.
            group_names: list of predefined group names for x-axis (size n)
            categories: list of question categories for y-axis (size m)
            alpha: float, alpha level for significance
            fig_filename: output figure filename
            dataset_prefix: title of the plot
        """
        data_cells = results[0]
        category_result = results[1]
        group_result = results[2]

        m, n = len(data_cells), len(data_cells[0])

        # Create x and y positions for categorical axes
        x_positions = np.arange(n + 1)
        y_positions = np.arange(m + 1)
        x_centers = (x_positions[:-1] + x_positions[1:]) / 2
        y_centers = (y_positions[:-1] + y_positions[1:]) / 2

        # Compute mean and labels
        mean_grid = np.empty((m, n))
        labels = [["" for _ in range(n)] for _ in range(m)]

        for i in range(m):
            for j in range(n):
                cell = data_cells[i][j]
                if cell is None or len(cell) == 0:
                    mean_grid[i, j] = 1 - alpha  # default color for N/A
                    labels[i][j] = "N/A\nCount: 0"
                else:
                    arr = np.array(cell)
                    mean = arr.mean()
                    std = arr.std()
                    mean_grid[i, j] = mean
                    labels[i][j] = f"{mean:.2f} ± {std:.2f}\nCount: {len(coverage_group[i][j])}"

        # Colormap: red → white → green
        cmap = mcolors.LinearSegmentedColormap.from_list(
            "red_white_green",
            [
                (0.0, (1.0, 0.2, 0.2)),
                ((0.3 - alpha) / 0.3, (1.0, 1.0, 0.95)),
                (1.0, (0.6, 1.0, 0.7))
            ]
        )
        norm = mcolors.Normalize(vmin=0.7, vmax=1.0, clip=True)

        # Plot
        fig, ax = plt.subplots()
        mesh = ax.pcolormesh(x_positions, y_positions, mean_grid, cmap=cmap, norm=norm,
                            edgecolors='k', linewidth=0.5, shading='auto')

        # Add text annotations in each cell
        for i in range(m):
            for j in range(n):
                ax.text(x_centers[j], y_centers[i], labels[i][j],
                        ha='center', va='center', fontsize=8)
    
        # Compute row pooled stats
        ytick_labels = []
        for i in range(m):
            cat_res = category_result[i]
            if cat_res is None or len(cat_res) == 0:
                ytick_labels.append("N/A\nCount: 0")
            else:
                arr = np.array(cat_res)
                mean = arr.mean()
                std = arr.std()
                color_tag = f"{mean:.2f}±{std:.2f}"
                ytick_labels.append(f"{categories[i]}\n{color_tag}")
        # Compute column pooled stats
        xtick_labels = []
        for j in range(n):
            group_res = group_result[j]
            if group_res is None or len(group_res) == 0:
                xtick_labels.append("N/A\nCount: 0")
            else:
                arr = np.array(group_res)
                mean = arr.mean()
                std = arr.std()
                color_tag = f"{mean:.2f}±{std:.2f}"
                xtick_labels.append(f"{group_names[j]}\n{color_tag}")

        # Axis labels and ticks
        ax.set_xticks(x_centers)
        ax.set_xticklabels(xtick_labels, rotation=45, ha='right')
        ax.set_yticks(y_centers)
        ax.set_yticklabels(ytick_labels)
        ax.set_xlabel('Predefined Groups')
        ax.set_ylabel('Question Categories')
        ax.set_title(dataset_prefix)
        ax.invert_yaxis()  # Highest category at top

        cbar = plt.colorbar(mesh, ax=ax)
        cbar.set_label("Mean Value (Color scale: Red → White → Green)")

        plt.savefig(fig_filename, bbox_inches="tight", dpi=800)
        plt.show()

    #Plotting part end

    def _compute_results(self, data, alphas, a, confidence_method, pre_defined_group=None):
        results = []
        for alpha in tqdm(alphas):
            results_for_alpha = [[], []]
            for i in range(len(data)):
                calibration_data = data[:i] + data[i + 1 :]
                test_data = data[i]
                threshold = self._compute_threshold(alpha, calibration_data, test_data, a, confidence_method, pre_defined_group)
                correctness, fraction_removed = self._evaluate_test_data(test_data, threshold, a, confidence_method)
                results_for_alpha[0].append(correctness)
                results_for_alpha[1].append(fraction_removed)
            results.append(results_for_alpha)
        return results

    def _compute_threshold(self, alpha, calibration_data, test_data, a, confidence_method, pre_defined_group=False):
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
                threshold = self._compute_threshold(alpha, calibration_data, test_data, a, confidence_method)
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
        if len(correctness_list) == 0:
            return 1
        return sum(correctness_list) / len(correctness_list)
    
    # def calibrate_factual_local(self, dataset_prefix, score_filepath, confidence_method, data, alpha, a, fig_filename, csv_filename, m, n):
    #     """
    #     m: the number of groups top subclaims scores divided into (y, row_idx)
    #     n: the number of groups true answers scores divided into (x, column_idx)
    #     """
    #     coverage_group, subclaim_edges, true_answer_edges = divide_coverage_group(score_filepath, m, n)
    #     results = self._process_partial_local(data, coverage_group, alpha, a, confidence_method, dataset_prefix)
    #     self.plot_stat_grid(coverage_group, results, true_answer_edges, subclaim_edges, alpha, fig_filename, dataset_prefix)

    # def prepare_result_holder(self, coverage_group, group_names):
    #     results = [[[] for _ in range(len(coverage_group[0]))] for _ in range(len(coverage_group))]
    #     if isinstance(group_names, list) and all(isinstance(name, str) for name in group_names):
    #         results = [{name: [] for name in group_names} for _ in range(len(coverage_group))]
    #     return results

    # def _process_partial_local(self, data, coverage_group, alpha, a, confidence_method, dataset_prefix, group_names = None):
    #     results = self.prepare_result_holder(coverage_group, group_names)
    #     for _ in tqdm(range(1000)):
    #         random.shuffle(data)
    #         split_index = len(data) // 2
    #         #split_index = 100
    #         calibration_data = data[:split_index]
    #         test_data = data[split_index:]

    #         threshold = self._compute_threshold(alpha, calibration_data, test_data, a, confidence_method)

    #         for i in range(len(coverage_group)):
    #             if isinstance(group_names, list) and all(isinstance(name, str) for name in group_names):
    #                 iterator = group_names
    #             else:
    #                 iterator = range(len(coverage_group[i]))

    #             for j in iterator:
    #                 if len(coverage_group[i][j]) == 0:
    #                     continue
    #                 bin_data = []
    #                 for query in coverage_group[i][j]:
    #                     for entry in test_data:
    #                         if query == entry["prompt"]:
    #                             bin_data.append(entry)
    #                 if len(bin_data) == 0:
    #                     continue
    #                 accepted_subclaim_list = self._get_accepted_subclaims(bin_data, threshold, confidence_method)
    #                 fraction_correct = self._compute_fraction_correct(accepted_subclaim_list, a)
    #                 results[i][j].append(fraction_correct)
    #     return results
    def divide_coverage_group(self, data, categories, group_names):
        coverage_group = [[[] for _ in range(len(group_names))] for _ in range(len(categories))]
        for i in range(len(data)):
            category_idx = categories.index(data[i]["category"])
            group_idx = group_names.index(data[i]["groups"][0])
            coverage_group[category_idx][group_idx].append(data[i]["prompt"])
        return coverage_group

    def set_result_holder(self, coverage_group):
        results = [[[] for _ in range(len(coverage_group[0]))] for _ in range(len(coverage_group))]
        results_per_row = [[] for _ in range(len(coverage_group))]
        results_per_col = [[] for _ in range(len(coverage_group[0]))]
        return results, results_per_row, results_per_col
    def prepare_calibration_data(self, data, split_index, max_size=100):
        if split_index <= max_size:
            calibration_data = data[:split_index]
        else:
            #randonly select n data points
            calibration_data = random.sample(data[:split_index], max_size)
        return calibration_data

    def _process_partial_local(self, data, coverage_group, alpha, a, confidence_method, dataset_prefix):
        results, results_per_row, results_per_col = self.set_result_holder(coverage_group)
        for _ in tqdm(range(2000)):
            random.shuffle(data)
            split_index = len(data) // 2
            calibration_data = data[:split_index]
            test_data = data[split_index:]

            threshold = self._compute_threshold(alpha, calibration_data, test_data, a, confidence_method)
            
            accepted_subclaim_list_per_row = [[] for _ in range(len(coverage_group))]
            accepted_subclaim_list_per_col = [[] for _ in range(len(coverage_group[0]))]

            for i in range(len(coverage_group)):
                for j in range(len(coverage_group[i])):
                    if len(coverage_group[i][j]) == 0:
                        continue
                    bin_data = []
                    for query in coverage_group[i][j]:
                        for entry in test_data:
                            if query == entry["prompt"]:
                                bin_data.append(entry)
                    if len(bin_data) == 0:
                        continue
                    accepted_subclaim_list = self._get_accepted_subclaims(bin_data, threshold, confidence_method)
                    accepted_subclaim_list_per_row[i].extend(accepted_subclaim_list)
                    accepted_subclaim_list_per_col[j].extend(accepted_subclaim_list)
                    fraction_correct = self._compute_fraction_correct(accepted_subclaim_list, a)
                    results[i][j].append(fraction_correct)
            for i in range(len(coverage_group)):
                fraction_correct = self._compute_fraction_correct(accepted_subclaim_list_per_row[i], a)
                results_per_row[i].append(fraction_correct)

            for j in range(len(coverage_group[0])):
                fraction_correct = self._compute_fraction_correct(accepted_subclaim_list_per_col[j], a)
                results_per_col[j].append(fraction_correct)
        return [results, results_per_row, results_per_col]
    
    
    def calibrate_factual_pergroup(self, dataset_prefix, confidence_method, data, alpha, a, fig_filename, csv_filename, categories, group_names):
        """
        categories: question type categoreis for y axis
        group_names: the group names for x axis
        """
        coverage_group = self.divide_coverage_group(data, categories, group_names)
        results = self._process_partial_local(data, coverage_group, alpha, a, confidence_method, dataset_prefix)
        self.plot_stat_grid(coverage_group, results, group_names, categories, alpha, fig_filename, dataset_prefix)


class WeightedConformalCalibration(ConformalCalibration):
    def __init__(self, embedding_model = "text-embedding-3-large"):
        super().__init__()
        self.prompt_embedding_cache = dc.Cache("data/cache/prompt_embedding")
        dotenv_path = os.path.join(os.getcwd(), '.env')
        load_dotenv(dotenv_path)
        self.client = OpenAI()
        self.distance_cache = {} #key is  sorted than concat prompts
        self.test_data_thresholds = {}
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
                threshold = self._compute_threshold(alpha, calibration_data, test_data, a, confidence_method)
                threshold_record[test_data["prompt"]] = threshold
                correctness, fraction_removed = self._evaluate_test_data(test_data, threshold, a, confidence_method)
                results_for_alpha[0].append(correctness)
                results_for_alpha[1].append(fraction_removed)
            results.append(results_for_alpha)
            with open(f"data/out/weighted/threshold/threshold_record_{alpha:.2f}.json", "w") as fopen:
                json.dump(threshold_record, fopen, indent=4)
        return results
    
    def _compute_threshold(self, alpha, calibration_data, test_data, a, confidence_method, pre_defined_group=None):
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
    
    def get_accepted_subclaims_full(self, alpha, test_data, calibration_data, a, confidence_method):
        accepted_subclaim_list = []
        for entry in test_data:
            threshold = self._compute_threshold(alpha, calibration_data, entry, a, confidence_method)
            if entry["prompt"] not in self.test_data_thresholds:
                self.test_data_thresholds[entry["prompt"]] = []
            self.test_data_thresholds[entry["prompt"]].append(f"{threshold:.5f}")
            accepted_subclaim_list.append(
                [subclaim for subclaim in entry["claims"]
                    if subclaim[confidence_method + "-score"] + subclaim.get("noise", 0) >= threshold]
            )
        return accepted_subclaim_list
    
    def _process_calibration(self, data, alphas, a, confidence_method):
        results = []
        for alpha in tqdm(alphas):
            results_for_alpha = [[], []]
            for _ in range(1000):
                random.shuffle(data)
                split_index = len(data) // 2
                calibration_data = data[:split_index]
                test_data = data[split_index:]
                accepted_subclaim_list = self.get_accepted_subclaims_full(alpha, test_data, calibration_data, a, confidence_method)
                fraction_correct = self._compute_fraction_correct(accepted_subclaim_list, a)
                results_for_alpha[0].append(1 - alpha)
                results_for_alpha[1].append(fraction_correct)
            results.append(results_for_alpha)
            with open(f"data/out/weighted/test_data_thresholds_alpha={alpha:.2f}.json", "w") as fopen:
                json.dump(self.test_data_thresholds, fopen)
            self.test_data_thresholds = {}
        return results
    
    def _load_threshold(self, filename):
        with open(filename, "r") as fopen:
            return json.load(fopen)

    def _process_partial_local(self, data, coverage_group, alpha, a, confidence_method, dataset_prefix, group_names = None):
        results, results_per_row, results_per_col = self.set_result_holder(coverage_group)
        for _ in tqdm(range(1000)):
            random.shuffle(data)
            split_index = len(data) // 2
            calibration_data = data[:split_index]
            test_data = data[split_index:]
            accepted_subclaim_list_per_row = [[] for _ in range(len(coverage_group))]
            accepted_subclaim_list_per_col = [[] for _ in range(len(coverage_group[0]))]

            for i in range(len(coverage_group)):
                for j in range(len(coverage_group[i])):
                    if len(coverage_group[i][j]) == 0:
                        continue
                    bin_data = []
                    for query in coverage_group[i][j]:
                        for entry in test_data:
                            if query == entry["prompt"]:
                                bin_data.append(entry)
                    if len(bin_data) == 0:
                        continue

                    accepted_subclaim_list = self.get_accepted_subclaims_full(alpha, bin_data, calibration_data, a, confidence_method)
                    fraction_correct = self._compute_fraction_correct(accepted_subclaim_list, a)
                    accepted_subclaim_list_per_row[i].extend(accepted_subclaim_list)
                    accepted_subclaim_list_per_col[j].extend(accepted_subclaim_list)
                    results[i][j].append(fraction_correct)
            with open(f"data/out/weighted/{dataset_prefix}_test_data_thresholds_alpha={alpha:.2f}.json", "w") as fopen:
                json.dump(self.test_data_thresholds, fopen)
            for i in range(len(coverage_group)):
                fraction_correct = self._compute_fraction_correct(accepted_subclaim_list_per_row[i], a)
                results_per_row[i].append(fraction_correct)
            for j in range(len(coverage_group[0])):
                fraction_correct = self._compute_fraction_correct(accepted_subclaim_list_per_col[j], a)
                results_per_col[j].append(fraction_correct)
        return [results, results_per_row, results_per_col]

class DistanceConformalCalibration(ConformalCalibration):
    def __init__(self, embedding_model = "text-embedding-3-large"):
        super().__init__()
        self.prompt_embedding_cache = dc.Cache("data/cache/prompt_embedding")
        dotenv_path = os.path.join(os.getcwd(), '.env')
        load_dotenv(dotenv_path)
        self.client = OpenAI()
        self.distance_cache = {} #key is  sorted than concat prompts, only use in _compute_threshold
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
    
    def _compute_threshold(self, alpha, calibration_data, test_data, a, confidence_method, pre_defined_group=None):
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
                    threshold = self._compute_threshold(alpha, calibration_data, entry, a, confidence_method)
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
            for _ in tqdm(range(2000)):
                group_threshold_cache = {}
                random.shuffle(data)
                calibration_data, test_data = self.split_each_group(data)
                accepted_subclaim_list = self._get_group_accepted_subclaims(test_data, calibration_data, alpha, a, confidence_method, group_threshold_cache)
                fraction_correct = self._compute_fraction_correct(accepted_subclaim_list, a)
                results_for_alpha[0].append(1 - alpha)
                results_for_alpha[1].append(fraction_correct)
            results.append(results_for_alpha)
        return results
    
    # def _process_partial_local(self, data, coverage_group, alpha, a, confidence_method, dataset_prefix, group_names = None):
    #     results = self.prepare_result_holder(coverage_group, group_names)
    #     group_threshold_cache = {}
    #     for _ in tqdm(range(1000)):
    #         random.shuffle(data)
    #         split_index = len(data) // 2
    #         #split_index = 100
    #         calibration_data, test_data = self.split_each_group(data)

    #         for i in range(len(coverage_group)):
    #             if isinstance(group_names, list) and all(isinstance(name, str) for name in group_names):
    #                 iterator = group_names
    #             else:
    #                 iterator = range(len(coverage_group[i]))

    #             for j in iterator:
    #                 if len(coverage_group[i][j]) == 0:
    #                     continue
    #                 bin_data = []
    #                 for query in coverage_group[i][j]:
    #                     for entry in test_data:
    #                         if query == entry["prompt"]:
    #                             bin_data.append(entry)
    #                 if len(bin_data) == 0:
    #                     continue
    #                 accepted_subclaim_list = self._get_group_accepted_subclaims(bin_data, calibration_data, alpha, a, confidence_method, group_threshold_cache)
    #                 fraction_correct = self._compute_fraction_correct(accepted_subclaim_list, a)
    #                 results[i][j].append(fraction_correct)
    #     return results
    def _process_partial_local(self, data, coverage_group, alpha, a, confidence_method, dataset_prefix):
        results, results_per_row, results_per_col = self.set_result_holder(coverage_group)
        group_threshold_cache = {}
        for _ in tqdm(range(2000)):
            random.shuffle(data)
            calibration_data, test_data = self.split_each_group(data)
            accepted_subclaim_list_per_row = [[] for _ in range(len(coverage_group))]
            accepted_subclaim_list_per_col = [[] for _ in range(len(coverage_group[0]))]
            for i in range(len(coverage_group)):
                for j in range(len(coverage_group[i])):
                    if len(coverage_group[i][j]) == 0:
                        continue
                    bin_data = []
                    for query in coverage_group[i][j]:
                        for entry in test_data:
                            if query == entry["prompt"]:
                                bin_data.append(entry)
                    if len(bin_data) == 0:
                        continue
                    accepted_subclaim_list = self._get_group_accepted_subclaims(bin_data, calibration_data, alpha, a, confidence_method, group_threshold_cache)
                    accepted_subclaim_list_per_row[i].extend(accepted_subclaim_list)
                    accepted_subclaim_list_per_col[j].extend(accepted_subclaim_list)
                    fraction_correct = self._compute_fraction_correct(accepted_subclaim_list, a)
                    results[i][j].append(fraction_correct)

            for i in range(len(coverage_group)):
                fraction_correct = self._compute_fraction_correct(accepted_subclaim_list_per_row[i], a)
                results_per_row[i].append(fraction_correct)
            for j in range(len(coverage_group[0])):
                fraction_correct = self._compute_fraction_correct(accepted_subclaim_list_per_col[j], a)
                results_per_col[j].append(fraction_correct)
        return [results, results_per_row, results_per_col]

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
                threshold = self._compute_threshold(alpha, None, test_data, a, confidence_method, pre_defined_group)
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
        self.faiss_manager = FAISSIndexManager({
            "strategy": "fixed_length",  # false
            "truncate_by": None,  # "\n"
            "chunk_size": 2000,
            "chunk_overlap": 25},
        dimension=3072,
        index_path="index_store/index.faiss",
        indice2fm_path="index_store/indice2fm.json",)
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
