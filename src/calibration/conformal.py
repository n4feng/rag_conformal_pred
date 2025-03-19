import os
import csv
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from src.calibration.base_calibration import ICalibration
from src.calibration.utils import compute_threshold
from src.calibration.utils import append_result_to_csv


CORRECT_ANNOTATIONS = ["S"]


class SplitConformalCalibration(ICalibration):
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
        x, y, yerr = self.process_conformal_removal_results(results)

        # write the results to csv file
        self._write_csv_header(csv_filename, alphas)
        append_result_to_csv(
            csv_filename=csv_filename,
            label=f"{self.dataset_name}_factual",
            y=y,
            yerr=yerr,
        )

        # plot the results
        print(f"Producing conformal plot: {fig_filename}")
        self.plot_conformal_removal_rate_by_alpha(
            self.dataset_name, x, y, yerr, a, fig_filename
        )

    def compute_conformal_results(
        self, data: list, alphas: np.ndarray, a: float, plot_group_results: bool = False
    ):

        results = {}
        for alpha in tqdm(alphas, desc="Computing conformal results"):
            # TODO add grouping
            groups = None
            if plot_group_results:
                # groups = test_data["groups"]
                raise NotImplementedError(
                    "Plotting by group is currently not supported."
                )

            thresholds = []
            correctness_list = []
            fraction_removed_list = []
            for _ in range(1000):
                random.shuffle(data)
                split_index = len(data) // 2
                calibration_data = data[:split_index]
                test_data = data[split_index:]

                assert (
                    len(calibration_data) != 0
                ), "Calibration data should not be empty"
                assert len(test_data) != 0, "Test data should not be empty"

                threshold = self._compute_threshold_by_group(
                    alpha, calibration_data, a, groups=groups
                )

                correctness, fraction_removed = self._evaluate_conformal_correctness(
                    test_data, threshold, a
                )
                thresholds.append(threshold)
                correctness_list.append(correctness)
                fraction_removed_list.append(fraction_removed)

            results[alpha] = {
                "threshold": thresholds,  # TODO: thresholds to be a list, same for correctness and faction removed
                "correctness": correctness_list,
                "fraction_removed": fraction_removed_list,
            }

        return results

    def process_conformal_removal_results(self, results: dict):
        """
        x: list of average correctness
        y: list of average fraction removed
        yerr: list of standard error of fraction removed
        """
        x, y, yerr = [], [], []
        for alpha, results_for_alpha in results.items():
            x_per_alpha = np.mean(
                results_for_alpha["correctness"]
            )  # correct retainment percentage at a specific alpha value, averaging over all test data points
            y_per_alpha = np.mean(
                results_for_alpha["fraction_removed"]
            )  # removal percentage at a specific alpha value, averaging over all test data points
            x.append(x_per_alpha)
            y.append(y_per_alpha)
            yerr.append(
                (
                    np.std(results_for_alpha["fraction_removed"])
                    * 1.96
                    / np.sqrt(len(results_for_alpha["fraction_removed"]))
                )
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

        # Ensure the directory exists
        os.makedirs(os.path.dirname(csv_filename), exist_ok=True)

        if not os.path.exists(csv_filename):
            with open(csv_filename, mode="w", newline="") as file:
                csv.writer(file).writerow(header)

    def _compute_threshold_by_group(
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

    def _evaluate_conformal_correctness(self, data: list, threshold: float, a: float):
        """
        Evaluates the performance of a conformal prediction model on test data.
        Parameters:
        data (list): A list of dictionaries, where each dictionary represents an entry with subclaims.
        threshold (float): The similarity score threshold to determine if a subclaim is correctly retained.
        a (float): The threshold for the correctly retained percentage to consider an entry as correctly retained.
        Returns:
        tuple: A tuple containing two lists:
            - correctly_retained (list): A list of boolean values indicating whether each entry is correctly retained.
            - fraction_removed (list): A list of floats representing the fraction of subclaims removed for each entry.
        """

        correctly_retained = []
        fraction_removed = []

        for entry in data:
            removal_count = 0
            correctly_retained_count = 0

            for subclaim in entry["subclaims"]:
                # Find similarity score
                similarity_score = subclaim.get("scores", {}).get("similarity", np.nan)
                if similarity_score >= threshold:
                    if (
                        subclaim.get("annotations", {}).get("gpt", "")
                        in CORRECT_ANNOTATIONS
                    ):
                        correctly_retained_count += 1
                else:
                    removal_count += 1

            total_subclaims = len(entry["subclaims"])
            retained_cnt = total_subclaims - removal_count

            # Calculate fraction of removed subclaims
            entry_removal_rate = (
                0 if total_subclaims == 0 else removal_count / total_subclaims
            )
            fraction_removed.append(
                entry_removal_rate
            )  # e.g. fraction_removed = [0.2, 0.5, 0.6, 0.2, 0.7] - one element per data entry

            # Calculate correctly retained rate
            correctly_retained_percentage = (
                correctly_retained_count / retained_cnt if retained_cnt > 0 else 1
            )
            correctly_retained.append(correctly_retained_percentage >= a)

        return correctly_retained, fraction_removed

    def plot_factual_removal(
        self, data, alphas, a, fig_filename, csv_filename, plot_group_results=False
    ):
        """ """
        print(f"Producing factual removal plot: {fig_filename}")
        x_values = np.linspace(1 - alphas[-1] - 0.05, 1 - alphas[0] + 0.03, 100)
        plt.figure(dpi=800)
        plt.plot(
            x_values, x_values, "--", color="gray", linewidth=2, label="Thrm 3.1 bounds"
        )

        results = self.compute_factual_results(data, alphas, a)
        x, y, yerr = self.process_factual_correctness_results(results)

        append_result_to_csv(
            csv_filename=csv_filename,
            label=f"{self.dataset_name}_conformal",
            y=y,
            yerr=yerr,
        )

        self.plot_factual_removal_rate_by_alpha(x, y, fig_filename)

        if plot_group_results:
            # self.plot_factual_group_results(results, csv_filename, x)
            raise NotImplementedError("Not implemented")

    def compute_factual_results(self, data, alphas, a):
        results = {}
        for alpha in tqdm(alphas, desc="Computing factual results"):
            thresholds = []
            correctness = []
            for _ in range(1000):
                random.shuffle(data)
                split_index = len(data) // 2
                calibration_data = data[:split_index]
                test_data = data[split_index:]

                assert (
                    len(calibration_data) != 0
                ), "Calibration data should not be empty"
                assert len(test_data) != 0, "Test data should not be empty"

                threshold = self._compute_threshold_by_group(
                    alpha, calibration_data, a, groups=None
                )
                fraction_correct = self._evaluate_factual_correctness(
                    test_data, threshold, a
                )
                thresholds.append(threshold)
                correctness.append(fraction_correct)

            results[alpha] = {
                "threshold": thresholds,
                "correctness": correctness,
                "factuality": 1 - alpha,
            }
        return results

    def process_factual_correctness_results(self, results: dict):
        """
        x: confidence level
        y: list of average factual correctness
        yerr: list of standard error of factual correctness
        """
        x, y, yerr = [], [], []
        for alpha, results_for_alpha in results.items():

            x.append(1 - alpha)
            y.append(np.mean(results_for_alpha["correctness"]))
            yerr.append(
                (
                    np.std(results_for_alpha["correctness"])
                    * 1.96
                    / np.sqrt(len(results_for_alpha["correctness"]))
                )
            )

        return x, y, yerr

    def plot_factual_removal_rate_by_alpha(self, x, y, fig_filename):
        plt.plot(x, y, label=self.dataset_name, linewidth=2)

        plt.xlabel(f"Target factuality (1 - {chr(945)})", fontsize=16)
        plt.ylabel("Empirical factuality", fontsize=16)
        plt.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), fontsize=10)
        plt.savefig(fig_filename, bbox_inches="tight", dpi=800)

    # def plot_factual_group_results(self, results, csv_filename, x):
    #     y_groups = {}
    #     y_groups_err = {}

    #     for result in results:
    #         for group, y_result in result[2].items():  # TODO: clarify why result[2]
    #             y_group = np.mean(y_result)
    #             y_group_err = np.std(y_result)
    #             y_groups.setdefault(group, []).append(y_group)
    #             y_groups_err.setdefault(group, []).append(y_group_err)

    #     for group, y_group in y_groups.items():
    #         label = f"{self.dataset_name}_conditional_{group}"
    #         # append_result_to_csv(csv_filename, label, y_group, y_groups_err[group])
    #         plt.plot(x, y_group, label=label, linewidth=2)

    def _evaluate_factual_correctness(self, data: list, threshold: float, a: float):
        """
        Evaluates the factual correctness of subclaims within the provided data.
        This function processes a list of data entries, each containing subclaims with similarity scores.
        It calculates the percentage of correctly retained subclaims based on a given threshold and
        compares it to a specified accuracy level `a`.
        Args:
            data (list): A list of dictionaries, where each dictionary represents an entry containing subclaims.
            threshold (float): The similarity score threshold above which subclaims are considered retained.
            a (float): The accuracy level to compare the correctly retained percentage against.
        Returns:
            float: The percentage of entries in the data that satisfy the correct level of accuracy `a`.
        """

        correctly_retained = []
        # Process each item in the list
        for entry in data:
            # Extract subclaims from each item
            retained_cnt = 0
            correctly_retained_count = 0
            for subclaim in entry["subclaims"]:

                # Extract the similarity score
                similarity_score = subclaim.get("scores", {}).get("similarity", np.nan)

                # Add the subcla 0-94=3im to the collection if similarity score is above threshold
                if similarity_score > threshold:
                    retained_cnt += 1
                    if (
                        subclaim.get("annotations", {}).get("gpt", "")
                        in CORRECT_ANNOTATIONS
                    ):
                        correctly_retained_count += 1

            # Calculate correctly retained rate
            correctly_retained_percentage = (
                correctly_retained_count / retained_cnt if retained_cnt > 0 else 1
            )
            correctly_retained.append(correctly_retained_percentage)

        correctness_list = [
            correctly_retained_percentage >= a
            for correctly_retained_percentage in correctly_retained
        ]
        # percentage of test data satisfying correct level of a
        return sum(correctness_list) / len(correctness_list)
