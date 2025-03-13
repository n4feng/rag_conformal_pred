from typing import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import csv
import random
from tqdm import tqdm

from src.calibration.base_calibration import ICalibration
from src.calibration.utils import compute_threshold
from src.calibration.utils import append_result_to_csv


CORRECT_ANNOTATIONS = ["Y", "S"]


class SplitConformal(ICalibration):
    """
    Implementation of standard conformal calibration.
    """

    def __init__(self, dataset_name: str, confidence_method: str):
        self.dataset_name = dataset_name
        self.confidence_method = confidence_method

    def plot_conformal_removal(
        self, data, alphas, a, fig_filename, csv_filename, plot_group_results=False
    ):

        # compute the correctness and fraction removed for each alpha
        results = self.compute_conformal_results(data, alphas, a, plot_group_results)
        x, y, yerr = self.process_conformal_results(results)

        # write the results to csv file
        self._write_csv_header(csv_filename, alphas)
        append_result_to_csv(csv_filename, self.dataset_name, y, yerr)

        # plot the results
        print(f"Producing conformal plot: {fig_filename}")
        self.plot_conformal_removal_rate_by_alpha(
            self.dataset_name, x, y, yerr, a, fig_filename
        )

    def compute_conformal_results(
        self, data: list, alphas: np.ndarray, a: float, plot_group_results: bool = False
    ):
        results = defaultdict(list)
        for alpha in alphas:
            results["alpha"].append(alpha)
            results_for_alpha = defaultdict(list)
            for i in range(len(data)):
                test_data = [data[i]]
                calibration_data = (
                    data[:i] + data[i + 1 :]
                )  # take the rest of the data as calibration data

                assert (
                    len(calibration_data) != 0
                ), "Calibration data should not be empty"
                assert len(test_data) != 0, "Test data should not be empty"

                # TODO add grouping
                groups = None
                if plot_group_results:
                    # groups = test_data["groups"]
                    raise NotImplementedError(
                        "Plotting by group is currently not supported."
                    )

                threshold = self.compute_threshold_by_group(
                    alpha, calibration_data, a, groups
                )
                correctness, fraction_removed = self.evaluate_conformal(
                    test_data, threshold, a
                )
                results_for_alpha["correctness"].append(correctness)
                results_for_alpha["fraction_removed"].append(fraction_removed)
            results["results_for_alpha"].append(results_for_alpha)

        return results

    def process_conformal_results(self, results: dict):
        """
        x: list of average correctness
        y: list of average fraction removed
        yerr: list of standard error of fraction removed
        """
        x, y, yerr = [], [], []
        for _, results_for_alpha in results.items():
            x.append(np.mean(results_for_alpha["correctness"]))
            y.append(np.mean(results_for_alpha["fraction_removed"]))
            yerr.append(
                np.std(results_for_alpha["fraction_removed"])
                * 1.96
                / np.sqrt(len(results_for_alpha["fraction_removed"]))
            )

        return x, y, yerr

    def plot_conformal_removal_rate_by_alpha(
        self, dataset_name, x, y, yerr, a, fig_filename
    ):

        plt.figure(dpi=800)
        plt.errorbar(x, y, yerr=yerr, label=dataset_name, linewidth=2)

        plt.xlabel(
            f"Fraction achieving avg factuality >= {a}"
            if a != 1
            else "Fraction of factual outputs"
        )
        plt.ylabel("Average percent removed")
        plt.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), fontsize=10)
        plt.title(f"Conformal Plots for {dataset_name} Datasets (a={a})", fontsize=20)
        plt.savefig(fig_filename, bbox_inches="tight")

    def _write_csv_header(self, csv_filename, alphas):
        target_factuality = [f"{(1-x):.2f}" for x in alphas][::-1]
        header = ["dataset"] + target_factuality
        with open(csv_filename, mode="w", newline="") as file:
            csv.writer(file).writerow(header)

    def compute_threshold_by_group(
        self, alpha, calibration_data, a, groups: list | None = None
    ):
        # TODO: clarify why take the min of the threshold of each group instead of returning a list of thresholds
        if groups:
            return min(
                compute_threshold(alpha, groups[group], a, self.confidence_method)
                for group in groups
            )
        else:
            # treat the whole data as calibration data
            return compute_threshold(alpha, calibration_data, a, self.confidence_method)

    def evaluate_conformal(self, test_data, threshold, a):s
        accepted_subclaims = [
            subclaim
            for subclaim in test_data["claims"]
            if subclaim[self.confidence_method + "-score"] + subclaim.get("noise", 0)
            >= threshold
        ]
        fraction_removed = (
            0
            if len(test_data["claims"]) == 0
            else 1 - len(accepted_subclaims) / len(test_data["claims"])
        )
        correctly_retained_percentage = (
            np.mean(
                [
                    subclaim["annotation"] in CORRECT_ANNOTATIONS
                    for subclaim in accepted_subclaims
                ]
            )
            >= a
            if accepted_subclaims
            else 1
        )
        return correctly_retained_percentage, fraction_removed

    def plot_factual_removal(
        self, data, alphas, a, fig_filename, csv_filename, plot_group_results=False
    ):
        """ """
        print(f"Producing calibration plot: {fig_filename}")
        x_values = np.linspace(1 - alphas[-1] - 0.05, 1 - alphas[0] + 0.03, 100)
        plt.plot(
            x_values, x_values, "--", color="gray", linewidth=2, label="Thrm 3.1 bounds"
        )

        results = self.compute_factual_results(data, alphas, a, self.confidence_method)
        x, y, yerr = self.process_conformal_results(results)

        append_result_to_csv(csv_filename, self.dataset_name, y, yerr)

        self.plot_factual_removal_rate_by_alpha(x, y, fig_filename)

        if plot_group_results:
            # self.plot_factual_group_results(results, csv_filename, x)
            raise NotImplementedError("Not implemented")

    def compute_factual_results(self, data, alphas, a):
        results = defaultdict(list)
        for alpha in alphas:
            results["alpha"].append(alpha)
            results_for_alpha = defaultdict(list)
            for _ in tqdm(range(1000), desc="Computing factual results"):
                random.shuffle(data)
                split_index = len(data) // 2
                calibration_data = data[:split_index]
                test_data = data[split_index:]
                threshold = compute_threshold(
                    alpha, calibration_data, a, self.confidence_method
                )
                fraction_correct = self._compute_factual_correctness(
                    test_data, threshold, a
                )
                results_for_alpha["factuality"].append(1 - alpha)
                results_for_alpha["correctness"].append(fraction_correct)
            results["results_for_alpha"].append(results_for_alpha)
        return results

    def plot_factual_removal_rate_by_alpha(self, x, y, fig_filename):
        plt.plot(x, y, label=self.dataset_name, linewidth=2)

        plt.xlabel(f"Target factuality (1 - {chr(945)})", fontsize=16)
        plt.ylabel("Empirical factuality", fontsize=16)
        plt.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), fontsize=10)
        plt.savefig(fig_filename, bbox_inches="tight", dpi=800)

    def plot_factual_group_results(self, results, csv_filename, x):
        y_groups = {}
        y_groups_err = {}

        for result in results:
            for group, y_result in result[2].items():  # TODO: clarify why result[2]
                y_group = np.mean(y_result)
                y_group_err = np.std(y_result)
                y_groups.setdefault(group, []).append(y_group)
                y_groups_err.setdefault(group, []).append(y_group_err)

        for group, y_group in y_groups.items():
            label = f"{self.dataset_name}_conditional_{group}"
            append_result_to_csv(csv_filename, label, y_group, y_groups_err[group])
            plt.plot(x, y_group, label=label, linewidth=2)

    def _compute_factual_correctness(self, data, threshold, a):
        accepted_subclaim_list = [
            [
                subclaim
                for subclaim in pt["claims"]
                if subclaim.get(self.confidence_method + "-score")
                + subclaim.get("noise", 0)
                >= threshold
            ]
            for pt in data
        ]
        entailed_fraction_list = [
            (
                np.mean(
                    [
                        subclaim["annotation"] in CORRECT_ANNOTATIONS
                        for subclaim in accepted_subclaims
                    ]
                )
                if accepted_subclaims
                else 1
            )
            for accepted_subclaims in accepted_subclaim_list
        ]
        correctness_list = [
            entailed_fraction >= a for entailed_fraction in entailed_fraction_list
        ]
        return sum(correctness_list) / len(correctness_list)
