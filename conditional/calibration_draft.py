import numpy as np
import json
import csv
import random
import numpy as np
import matplotlib.pyplot as plt
from rag.scorer.wikitexts_embedding import WikitextsDocumentScorer
from tqdm import tqdm
from math import ceil

#global variables
wikiEmbedding = WikitextsDocumentScorer()
METHOD_SUPPORT_CONDITION = ['similarity','gpt']
CORRECT_ANNOTATIONS = ["Y", "S"]
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

def create_correctness_vs_removed_plot(
    dataset_prefixs, confidence_method, datasets, alphas, a, fig_filename, csv_filename, is_conformal = True, is_conditional=False
):
    """
    Creates leave-one-out conformal plots for all datasets in dataset_prefixs.
    """
    print(f"Producing conformal plot: {fig_filename}")
    plt.figure(dpi=800)
    target_factuality = [f"{(1-x):.2f}" for x in alphas]
    target_factuality.reverse()
    header = ["dataset"] + target_factuality

    # Write to CSV
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
    
    #Conformal Prediction
    if is_conformal:
        for dataset_prefix in tqdm(dataset_prefixs):
            data = datasets[dataset_prefix]

            results = calculate_correctness_and_removal(
                data, alphas, a, confidence_method
            )

            x = [np.mean(results_for_alpha[0]) for results_for_alpha in results]
            y = [np.mean(results_for_alpha[1]) for results_for_alpha in results]

            # Add standard error.
            yerr = [
                np.std(results_for_alpha[1]) * 1.96 / np.sqrt(len(results_for_alpha[1]))
                for results_for_alpha in results
            ]
            label = dataset_prefix
            append_result_to_csv(csv_filename, label, y, yerr)
            plt.errorbar(x, y, yerr=yerr, label=label, linewidth=2)

    #Conditional Conformal Perdiction
    if is_conditional:
        for dataset_prefix in tqdm(dataset_prefixs):
            if confidence_method not in METHOD_SUPPORT_CONDITION:
                continue
            data = datasets[dataset_prefix]

            results = calculate_correctness_and_removal(
                data, alphas, a, confidence_method, True
            )

            x = [np.mean(results_for_alpha[0]) for results_for_alpha in results]
            y = [np.mean(results_for_alpha[1]) for results_for_alpha in results]

            # Add standard error.
            yerr = [
                np.std(results_for_alpha[1]) * 1.96 / np.sqrt(len(results_for_alpha[1]))
                for results_for_alpha in results
            ]
            label = dataset_prefix + '_conditional'
            append_result_to_csv(csv_filename, label, y, yerr)
            plt.errorbar(x, y, yerr=yerr, label=label, linewidth=2)

    # Plot base factuality point for the last dataset in the loop.
    x_point = x[-1]
    y_point = y[-1]
    point_size = 235
    plt.scatter(
        x_point,
        y_point,
        color="black",
        marker="*",
        s=point_size,
        label="Base factuality",
        zorder=1000,
    )

    font_size = 16
    legend_font_size = 10
    plt.title(f"Conformal Plots for {dataset_prefixs} Datasets (a={a})", fontsize=font_size + 4)
    plt.xlabel(
        f"Fraction achieving avg factuality >= {a}" if a != 1 else "Fraction of factual outputs",
        fontsize=font_size,
    )
    plt.ylabel("Average percent removed", fontsize=font_size)

    legend = plt.legend(
        loc="upper left", bbox_to_anchor=(0.02, 0.98), fontsize=legend_font_size
    )
    legend.get_title().set_fontsize(legend_font_size)
    plt.savefig(fig_filename, bbox_inches="tight")


def calculate_correctness_and_removal(
    data, alphas, a, confidence_method, is_conditional=False
):
    """
    Calculates correctness and fraction removed for a dataset over a range of alphas.

    Args:
        data (list): The dataset, where each entry contains claims and annotations.
        predifned_group (list): List of predefined groups for threshold computation. only used when is_conditional is True.
        alphas (list): List of alpha values for threshold computation.
        a (float): Minimum entailed fraction threshold for correctness.
        confidence_method (str): The method used to compute confidence.

    Returns:
        list: Results containing average correctness and fraction removed for each alpha.
    """
    pre_defined_group = {}
    if is_conditional and confidence_method in METHOD_SUPPORT_CONDITION:
        for item in data:
            groups = item['groups']
            for group in groups:
                if group in pre_defined_group:
                    pre_defined_group[group].append(item)
                else:
                    pre_defined_group[group] = [item]
    results = []  # first indexes into alpha, then list of (correct, frac_removed)_i
    for alpha in alphas:
        results_for_alpha = [[], []]        
        for i in range(len(data)):
            # Leave-one-out calibration data
            calibration_data = data[:i] + data[i + 1 :]
            test_data = data[i]

            # Compute the threshold using the provided function
            threshold = 1.0

            if is_conditional and confidence_method in METHOD_SUPPORT_CONDITION:
                #note that test data should already exist in the group
                for group in test_data['groups']:
                    group_data = pre_defined_group[group]
                    threshold = min(threshold,compute_threshold(
                        alpha, group_data, a, confidence_method
                    ))
            else:
                threshold = compute_threshold(
                    alpha, calibration_data, a, confidence_method
                )
            # Determine accepted subclaims
            accepted_subclaims = [
                subclaim
                for subclaim in test_data["claims"]
                if subclaim[confidence_method + "-score"] + subclaim.get("noise", 0)
                >= threshold
            ]
            total_claim = len(test_data["claims"])
            fraction_removed = (
                0 if total_claim == 0 else 1 - len(accepted_subclaims) / total_claim
            )
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
            correctness = entailed_fraction >= a
            results_for_alpha[0].append(correctness)
            results_for_alpha[1].append(fraction_removed)
            
        print(f"processing for alpha {alpha} done")
        results.append(results_for_alpha)
    return results

def create_calibration_plot(
    dataset_prefix, confidence_method, data, alphas, a, fig_filename, csv_filename, is_conformal=True, is_conditional=False
):
    """
    Creates calibration plot.
    """
    print(f"Producing calibration plot: {fig_filename}")
    fig, ax = plt.subplots(figsize=(6, 4))

    #Conformal Prediction
    split_index = len(data) // 2

    x_values = np.linspace(1-alphas[-1] - 0.05, 1-alphas[0]+0.03, 100)

    # Plot lower bound.
    y_values = x_values
    plt.plot(
        x_values, y_values, "--", color="gray", linewidth=2, label="Thrm 3.1 bounds"
    )
    target_factuality = [f"{(1-x):.2f}" for x in alphas]
    target_factuality.reverse()
    header = ["dataset"] + target_factuality

    # Write to CSV
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
    
    if is_conformal:
        results = calculate_real_calibration_factuality(
            data, alphas, a, confidence_method
        )
        print(results)
        print("-----------------------------------------------------------")

        x = [np.mean(results_for_alpha[0]) for results_for_alpha in results]
        y = [np.mean(results_for_alpha[1]) for results_for_alpha in results]
        yerr = [np.std(results_for_alpha[1]) for results_for_alpha in results]

        label = dataset_prefix
        append_result_to_csv(csv_filename, label, y, yerr)
        #print(x)
        #print(y)
        # plt.fill_between(np.array(x), np.array(y) - np.array(yerr), np.array(y) + np.array(yerr), color="#ADD8E6")

        # Plot upper bound
        #y_values = x_values + 1 / (split_index + 1)
        #plt.plot(x_values, y_values, "--", color="gray", linewidth=2)
        plt.plot(x, y, label=dataset_prefix, linewidth=2)

    
    if is_conditional and confidence_method in METHOD_SUPPORT_CONDITION:
        results = calculate_real_calibration_factuality(
            data, alphas, a, confidence_method, True
        )
        print("Results:", results)
        x = [np.mean(results_for_alpha[0]) for results_for_alpha in results]
        y = [np.mean(results_for_alpha[1]) for results_for_alpha in results]
        yerr = [np.std(results_for_alpha[1]) for results_for_alpha in results]

        label = dataset_prefix + '_conditional'
        append_result_to_csv(csv_filename, label, y, yerr)

        #print(x)
        #print(y)
        # plt.fill_between(np.array(x), np.array(y) - np.array(yerr), np.array(y) + np.array(yerr), color="#ADD8E6")
        plt.plot(x, y, label=dataset_prefix + '_conditional', linewidth=2)

    plt.xlabel(f"Target factuality (1 - {chr(945)})", fontsize=16)
    plt.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), fontsize=10)
    plt.ylabel("Empirical factuality", fontsize=16)
    plt.savefig(fig_filename, bbox_inches="tight", dpi=800)

def calculate_real_calibration_factuality(
    data, alphas, a, confidence_method, is_conditional=False
):
    """
    Calculates fraction of supported sub-claims in conformal set for a dataset over a range of alphas.

    Args:
        data (list): The dataset, where each entry contains claims and annotations.
        alphas (list): List of alpha values for threshold computation.
        a (float): Minimum entailed fraction threshold for correctness.
        confidence_method (str): The method used to compute confidence.

    Returns:
        list: Results containing real conformaled sub-claims factuality fraction for each alpha.
    """
    results = []  # first indexes into alpha. then list of (correct, frac_removed)_i
    for alpha in tqdm(alphas):
        results_for_alpha = [[], []]
        for i in range(1000):
            group_threshold_cache = {}
            # Randomly shuffle the data
            random.shuffle(data)

            # Split the data into two equal parts
            split_index = len(data) // 2
            calibration_data = data[:split_index]
            test_data = data[split_index:]

            if is_conditional and confidence_method in METHOD_SUPPORT_CONDITION:
                calibration_data, test_data = split_each_group(data)

            # regroup in calibration data:
            pre_defined_group = {}
            if is_conditional and confidence_method in METHOD_SUPPORT_CONDITION:
                
                for item in data:
                    groups = item['groups']
                    for group in groups:
                        if group in pre_defined_group:
                            pre_defined_group[group].append(item)
                        else:
                            pre_defined_group[group] = [item]

            threshold = compute_threshold(alpha, calibration_data, a, confidence_method)
            accepted_subclaim_list = []
            for test_data_point in test_data:
                if is_conditional and confidence_method in METHOD_SUPPORT_CONDITION:
                    for group in test_data_point['groups']:
                        #we should not use test data as in reality we won't know the annotation of it
                        threshold = 1.0 #reset threshold
                        group_tresh = 1.0
                        if group in group_threshold_cache:
                            group_tresh = group_threshold_cache[group]
                        else:
                            group_tresh = compute_threshold(alpha, pre_defined_group[group], a, confidence_method)
                            group_threshold_cache[group] = group_tresh
                        threshold = min(threshold, group_tresh)
                
                accepted_subclaims = [
                    subclaim
                    for subclaim in test_data_point["claims"]
                    if subclaim[confidence_method + "-score"] + subclaim.get("noise", 0) >= threshold
                ]
                accepted_subclaim_list.append(accepted_subclaims)

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
            fraction_correct = sum(correctness_list) / len(correctness_list)
            results_for_alpha[0].append(1 - alpha)
            results_for_alpha[1].append(fraction_correct)
        #print(f"processing for alpha {alpha} done")
        results.append(results_for_alpha)
    return results

def create_calibration_plot_with_subgroup(
    dataset_prefix, confidence_method, data, alphas, a, fig_filename, csv_filename, is_conformal=True, is_conditional=False
):
    """
    Creates calibration plot.
    """
    print(f"Producing calibration plot: {fig_filename}")
    fig, ax = plt.subplots(figsize=(6, 4))

    #Conformal Prediction
    split_index = len(data) // 2

    x_values = np.linspace(1-alphas[-1] - 0.05, 1-alphas[0]+0.03, 100)

    # Plot lower bound.
    y_values = x_values
    plt.plot(
        x_values, y_values, "--", color="gray", linewidth=2, label="Thrm 3.1 bounds"
    )
    target_factuality = [f"{(1-x):.2f}" for x in alphas]
    target_factuality.reverse()
    header = ["dataset"] + target_factuality

    # Write to CSV
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
    
    if is_conformal:
        results = calculate_real_calibration_factuality_with_subgroup(
            data, alphas, a, confidence_method
        )

        x = [np.mean(results_for_alpha[0]) for results_for_alpha in results]
        y = [np.mean(results_for_alpha[1]) for results_for_alpha in results]
        yerr = [np.std(results_for_alpha[1]) for results_for_alpha in results]

        label = dataset_prefix
        append_result_to_csv(csv_filename, label, y, yerr)
        #print(x)
        #print(y)
        # plt.fill_between(np.array(x), np.array(y) - np.array(yerr), np.array(y) + np.array(yerr), color="#ADD8E6")

        # Plot upper bound
        #y_values = x_values + 1 / (split_index + 1)
        #plt.plot(x_values, y_values, "--", color="gray", linewidth=2)
        plt.plot(x, y, label=dataset_prefix, linewidth=2)

    
    if is_conditional and confidence_method in METHOD_SUPPORT_CONDITION:
        results = calculate_real_calibration_factuality_with_subgroup(
            data, alphas, a, confidence_method, True
        )

        x = [np.mean(results_for_alpha[0]) for results_for_alpha in results]
        y = [np.mean(results_for_alpha[1]) for results_for_alpha in results]
        yerr = [np.std(results_for_alpha[1]) for results_for_alpha in results]

        label = dataset_prefix + '_conditional'
        append_result_to_csv(csv_filename, label, y, yerr)

        #print(x)
        #print(y)
        # plt.fill_between(np.array(x), np.array(y) - np.array(yerr), np.array(y) + np.array(yerr), color="#ADD8E6")
        plt.plot(x, y, label=dataset_prefix + '_conditional', linewidth=2)
        
        y_groups = {}
        y_groups_err = {}
        for result in results:
            for group, y_result in result[2].items():
                y_group = np.mean(y_result)
                y_group_err = np.std(y_result)
                if group not in y_groups:
                    y_groups[group] = []
                    y_groups_err[group] = []
                y_groups[group].append(y_group)
                y_groups_err[group].append(y_group_err)
        
        for group, y_group in y_groups.items():
            label = f"{dataset_prefix}_conditional_{group}"
            append_result_to_csv(csv_filename, label, y_group, y_groups_err[group])
            plt.plot(x, y_group, label=label, linewidth=2)


    plt.xlabel(f"Target factuality (1 - {chr(945)})", fontsize=16)
    plt.legend(loc="upper left", bbox_to_anchor=(0.02, 0.98), fontsize=10)
    plt.ylabel("Empirical factuality", fontsize=16)
    plt.savefig(fig_filename, bbox_inches="tight", dpi=800)

def split_each_group(data, calibrate_range=0.5):
    """
    Split each group into a separate entry.
    """
    group_data = {}
    calibration_data = []
    test_data = []
    for entry in data:
        groups = entry["groups"]
        #split by first group as default group
        if groups[0] in group_data:
            group_data[groups[0]].append(entry)
        else:
            group_data[groups[0]] = [entry]
    for group, group_entries in group_data.items():
        split_index = ceil(len(group_entries) * calibrate_range)
        calibration_data.extend(group_entries[:split_index])
        test_data.extend(group_entries[split_index:])
    return calibration_data, test_data

def calculate_real_calibration_factuality_with_subgroup(
    data, alphas, a, confidence_method, is_conditional=False
):
    """
    Calculates fraction of supported sub-claims in conformal set for a dataset over a range of alphas.

    Args:
        data (list): The dataset, where each entry contains claims and annotations.
        alphas (list): List of alpha values for threshold computation.
        a (float): Minimum entailed fraction threshold for correctness.
        confidence_method (str): The method used to compute confidence.

    Returns:
        list: Results containing real conformaled sub-claims factuality fraction for each alpha.
    """
    results = []  # first indexes into alpha. then list of (correct, frac_removed)_i
    for alpha in tqdm(alphas):
        results_for_alpha = [[], [], {}]
        for i in range(1000):
            group_threshold_cache = {}
            # Randomly shuffle the data
            random.shuffle(data)

            # Split the data into two equal parts
            split_index = len(data) // 2
            calibration_data = data[:split_index]
            test_data = data[split_index:]

            if is_conditional and confidence_method in METHOD_SUPPORT_CONDITION:
                calibration_data, test_data = split_each_group(data)

            # regroup in calibration data:
            pre_defined_group = {}
            if is_conditional and confidence_method in METHOD_SUPPORT_CONDITION:
                
                for item in data:
                    groups = item['groups']
                    for group in groups:
                        if group in pre_defined_group:
                            pre_defined_group[group].append(item)
                        else:
                            pre_defined_group[group] = [item]

            threshold = compute_threshold(alpha, calibration_data, a, confidence_method)
            accepted_subclaim_list = []
            accepted_subclaim_list_pergroup = {}
                
            for test_data_point in test_data:
                if is_conditional and confidence_method in METHOD_SUPPORT_CONDITION:
                    for group in test_data_point['groups']:
                        #we should not use test data as in reality we won't know the annotation of it
                        threshold = 1.0 #reset threshold
                        group_tresh = 1.0
                        if group in group_threshold_cache:
                            group_tresh = group_threshold_cache[group]
                        else:
                            group_tresh = compute_threshold(alpha, pre_defined_group[group], a, confidence_method)
                            accepted_subclaim_list_pergroup[group] = []
                            group_threshold_cache[group] = group_tresh
                        threshold = min(threshold, group_tresh)
                
                accepted_subclaims = [
                    subclaim
                    for subclaim in test_data_point["claims"]
                    if subclaim[confidence_method + "-score"] + subclaim.get("noise", 0) >= threshold
                ]
                accepted_subclaim_list.append(accepted_subclaims)

                if is_conditional and confidence_method in METHOD_SUPPORT_CONDITION:
                    for group in test_data_point['groups']:
                        accepted_subclaim_list_pergroup[group].append([
                            subclaim
                            for subclaim in test_data_point["claims"]
                            if subclaim[confidence_method + "-score"] + subclaim.get("noise", 0) >= threshold
                        ])
                

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
            fraction_correct = sum(correctness_list) / len(correctness_list)
            results_for_alpha[0].append(1 - alpha)
            results_for_alpha[1].append(fraction_correct)

            if is_conditional and confidence_method in METHOD_SUPPORT_CONDITION:
                for group in accepted_subclaim_list_pergroup:
                    entailed_fraction_list_pergroup = [
                        (
                            np.mean(
                                [
                                    subclaim["annotation"] in CORRECT_ANNOTATIONS
                                    for subclaim in accepted_subclaims_pergroup
                                ]
                            )
                            if accepted_subclaims_pergroup
                            else 1
                        )
                        for accepted_subclaims_pergroup in accepted_subclaim_list_pergroup[group]
                    ]
                    correctness_list_pergroup = [
                        entailed_fraction >= a for entailed_fraction in entailed_fraction_list_pergroup
                    ]

                    fraction = sum(correctness_list_pergroup) / len(correctness_list_pergroup)
                    if group not in results_for_alpha[2]:
                        results_for_alpha[2][group] = []
                    results_for_alpha[2][group].append(fraction)
        #print(f"processing for alpha {alpha} done")
        results.append(results_for_alpha)
    return results
def create_correctness_vs_removed_plot_with_subgroup(
    dataset_prefixs, confidence_method, datasets, alphas, a, fig_filename, csv_filename, is_conformal = True, is_conditional=False
):
    """
    Creates leave-one-out conformal plots for all datasets in dataset_prefixs.
    """
    print(f"Producing conformal plot: {fig_filename}")
    plt.figure(dpi=800)
    target_factuality = [f"{(1-x):.2f}" for x in alphas]
    target_factuality.reverse()
    header = ["dataset"] + target_factuality

    # Write to CSV
    with open(csv_filename, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(header)
    
    #Conformal Prediction
    if is_conformal:
        for dataset_prefix in tqdm(dataset_prefixs):
            data = datasets[dataset_prefix]

            results = calculate_correctness_and_removal_with_subgroup(
                data, alphas, a, confidence_method
            )

            x = [np.mean(results_for_alpha[0]) for results_for_alpha in results]
            y = [np.mean(results_for_alpha[1]) for results_for_alpha in results]

            # Add standard error.
            yerr = [
                np.std(results_for_alpha[1]) * 1.96 / np.sqrt(len(results_for_alpha[1]))
                for results_for_alpha in results
            ]
            label = dataset_prefix
            append_result_to_csv(csv_filename, label, y, yerr)
            plt.errorbar(x, y, yerr=yerr, label=label, linewidth=2)

    #Conditional Conformal Perdiction
    if is_conditional:
        for dataset_prefix in tqdm(dataset_prefixs):
            if confidence_method not in METHOD_SUPPORT_CONDITION:
                continue
            data = datasets[dataset_prefix]

            results = calculate_correctness_and_removal_with_subgroup(
                data, alphas, a, confidence_method, True
            )

            x = [np.mean(results_for_alpha[0]) for results_for_alpha in results]
            y = [np.mean(results_for_alpha[1]) for results_for_alpha in results]

            # Add standard error.
            yerr = [
                np.std(results_for_alpha[1]) * 1.96 / np.sqrt(len(results_for_alpha[1]))
                for results_for_alpha in results
            ]
            label = dataset_prefix + '_conditional'
            append_result_to_csv(csv_filename, label, y, yerr)
            plt.errorbar(x, y, yerr=yerr, label=label, linewidth=2)
            y_groups = {}
            y_groups_err = {}
            for result in results:
                for group, y_result in result[2].items():
                    y_group = np.mean(y_result)
                    y_group_err = np.std(y_result) * 1.96 / np.sqrt(len(y_result))
                    if group not in y_groups:
                        y_groups[group] = []
                        y_groups_err[group] = []
                    y_groups[group].append(y_group)
                    y_groups_err[group].append(y_group_err)
            
            for group, y_group in y_groups.items():
                label = f"{dataset_prefix}_conditional_{group}"
                append_result_to_csv(csv_filename, label, y_group, y_groups_err[group])
                plt.errorbar(x, y_group, yerr=y_groups_err[group], label=label, linewidth=2)

    # Plot base factuality point for the last dataset in the loop.
    x_point = x[-1]
    y_point = y[-1]
    point_size = 235
    plt.scatter(
        x_point,
        y_point,
        color="black",
        marker="*",
        s=point_size,
        label="Base factuality",
        zorder=1000,
    )

    font_size = 16
    legend_font_size = 10
    plt.title(f"Conformal Plots for {dataset_prefixs} Datasets (a={a})", fontsize=font_size + 4)
    plt.xlabel(
        f"Fraction achieving avg factuality >= {a}" if a != 1 else "Fraction of factual outputs",
        fontsize=font_size,
    )
    plt.ylabel("Average percent removed", fontsize=font_size)

    legend = plt.legend(
        loc="upper left", bbox_to_anchor=(0.02, 0.98), fontsize=legend_font_size
    )
    legend.get_title().set_fontsize(legend_font_size)
    plt.savefig(fig_filename, bbox_inches="tight")


def calculate_correctness_and_removal_with_subgroup(
    data, alphas, a, confidence_method, is_conditional=False
):
    """
    Calculates correctness and fraction removed for a dataset over a range of alphas.

    Args:
        data (list): The dataset, where each entry contains claims and annotations.
        predifned_group (list): List of predefined groups for threshold computation. only used when is_conditional is True.
        alphas (list): List of alpha values for threshold computation.
        a (float): Minimum entailed fraction threshold for correctness.
        confidence_method (str): The method used to compute confidence.

    Returns:
        list: Results containing average correctness and fraction removed for each alpha.
    """
    pre_defined_group = {}
    if is_conditional and confidence_method in METHOD_SUPPORT_CONDITION:
        for item in data:
            groups = item['groups']
            for group in groups:
                if group in pre_defined_group:
                    pre_defined_group[group].append(item)
                else:
                    pre_defined_group[group] = [item]
    results = []  # first indexes into alpha, then list of (correct, frac_removed)_i
    for alpha in alphas:
        results_for_alpha = [[], [], {}]        
        for i in range(len(data)):
            # Leave-one-out calibration data
            calibration_data = data[:i] + data[i + 1 :]
            test_data = data[i]

            # Compute the threshold using the provided function
            threshold = 1.0

            if is_conditional and confidence_method in METHOD_SUPPORT_CONDITION:
                #note that test data should already exist in the group
                for group in test_data['groups']:
                    group_data = pre_defined_group[group]
                    threshold = min(threshold,compute_threshold(
                        alpha, group_data, a, confidence_method
                    ))
            else:
                threshold = compute_threshold(
                    alpha, calibration_data, a, confidence_method
                )
            # Determine accepted subclaims
            accepted_subclaims = [
                subclaim
                for subclaim in test_data["claims"]
                if subclaim[confidence_method + "-score"] + subclaim.get("noise", 0)
                >= threshold
            ]
            total_claim = len(test_data["claims"])
            fraction_removed = (
                0 if total_claim == 0 else 1 - len(accepted_subclaims) / total_claim
            )
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
            correctness = entailed_fraction >= a
            results_for_alpha[0].append(correctness)
            results_for_alpha[1].append(fraction_removed)
            if is_conditional and confidence_method in METHOD_SUPPORT_CONDITION:
                for group in test_data['groups']:
                    if group not in results_for_alpha[2]:
                        results_for_alpha[2][group] = []
                    results_for_alpha[2][group].append(fraction_removed)
            
        print(f"processing for alpha {alpha} done")
        results.append(results_for_alpha)
    return results