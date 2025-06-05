# -*- coding: utf-8 -*-
import time
import pandas as pd
import numpy as np
import json
import argparse
import optuna

from src.calibration.conformal import SplitConformalCalibration


def propose_new_score(data, beta0, beta1, beta2, beta3):
    for x in data:
        score4 = len(x["retrieved_docs"])
        for subclaim in x["subclaims"]:
            score1 = subclaim["scores"]["doc_claim_cosine_similarity"]
            score2 = subclaim["scores"]["query_claim_cosine_similarity"]
            score3 = subclaim["scores"]["relavance_max"]
            subclaim["scores"]["proposed_score"] = (
                beta0 * score1 + beta1 * score2 + beta2 * score3 + beta3 * score4
            )

    return data


def objective(
    trial: optuna.Trial, calib_data, alpha: float = 0.05, a: float = 1
) -> float:

    alpha_list = [alpha] if isinstance(alpha, float) else alpha

    beta0 = trial.suggest_float("beta0", -100, 100)
    beta1 = trial.suggest_float("beta1", -100, 100)
    beta2 = trial.suggest_float("beta2", -100, 100)
    beta3 = trial.suggest_float("beta3", -100, 100)

    data = propose_new_score(
        data=calib_data, beta0=beta0, beta1=beta1, beta2=beta2, beta3=beta3
    )

    conformal = SplitConformalCalibration(
        dataset_name="PopQA",
        score_types=["proposed_score"],
    )
    conformal_results = conformal.compute_conformal_results(data, alpha_list, a)
    print(
        f"Trial: {trial.number}, beta0: {beta0}, beta1: {beta1}, beta2: {beta2}, beta3: {beta3}, alpha: {alpha},"
        # f"Conformal Results: {conformal_results}"
    )
    removal_rate = np.mean(
        conformal_results["proposed_score"][alpha]["fraction_removed"]
    )

    return removal_rate


class CustomEarlyStoppingCallback:
    def __init__(self, patience=10, min_improvement=0.01):
        self.patience = patience
        self.min_improvement = min_improvement
        self.best_value = None
        self.trials_without_improvement = 0

    def __call__(self, study, trial):
        # Get current best value
        current_best = study.best_value

        # Check if this is the first trial or an improvement
        if self.best_value is None:
            self.best_value = current_best
            self.trials_without_improvement = 0
        elif study.direction == optuna.study.StudyDirection.MINIMIZE:
            if current_best < self.best_value - self.min_improvement:
                self.best_value = current_best
                self.trials_without_improvement = 0
            else:
                self.trials_without_improvement += 1
        else:  # MAXIMIZE
            if current_best > self.best_value + self.min_improvement:
                self.best_value = current_best
                self.trials_without_improvement = 0
            else:
                self.trials_without_improvement += 1

        # Stop if patience exceeded
        if self.trials_without_improvement >= self.patience:
            study.stop()
            print(
                f"Early stopping triggered after {self.trials_without_improvement} trials without improvement"
            )


class TimeBasedEarlyStopping:
    def __init__(self, max_duration_seconds):
        self.max_duration = max_duration_seconds
        self.start_time = None

    def __call__(self, study, trial):
        if self.start_time is None:
            self.start_time = time.time()

        elapsed = time.time() - self.start_time
        if elapsed > self.max_duration:
            study.stop()
            print(f"Time limit reached: {elapsed:.1f} seconds")


def run_hpo(
    data,
    objective_func,
    alpha: float = 0.05,
    n_trials: int = 100,
    seed: int = 42,
    verbose: bool = True,
):
    """Run hyperparameter optimization."""
    # This function is just a placeholder to illustrate where the HPO would be run.
    # The actual implementation would depend on the specific requirements of the task.

    if not verbose:
        # Suppress Optuna's logging outpu
        optuna.logging.set_verbosity(optuna.logging.WARNING)

    sampler = optuna.samplers.TPESampler(
        seed=seed,
        n_startup_trials=10,
        # multivariate=True,
        warn_independent_sampling=True,
    )
    # sampler = optuna.samplers.RandomSampler(seed=seed)
    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
    )

    study.enqueue_trial({"beta0": 1, "beta1": 0, "beta2": 0, "beta3": 0})
    study.enqueue_trial({"beta0": 0, "beta1": 1, "beta2": 0, "beta3": 0})
    study.enqueue_trial({"beta0": 0, "beta1": 0, "beta2": 1, "beta3": 0})

    # Stop if no improvement for 100 consecutive trials
    early_stopping = CustomEarlyStoppingCallback(patience=50, min_improvement=0.0005)
    time_limit = TimeBasedEarlyStopping(max_duration_seconds=30 * 60)

    study.optimize(
        lambda trial: objective_func(trial, data, alpha=alpha),
        n_trials=n_trials,
        callbacks=[early_stopping, time_limit],
        show_progress_bar=True,
    )

    return study.best_params, study.best_value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run hyperparameter optimization.")
    parser.add_argument(
        "--file_path",
        type=str,
        required=True,
        help="Path to the input JSON file containing the data.",
    )
    parser.add_argument(
        "--n_trials", type=int, default=1000, help="Number of trials for HPO."
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for conformal calibration.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Enable verbose output."
    )
    args = parser.parse_args()

    with open(args.file_path, "r") as file_path:
        # Load the JSON data
        data = json.load(file_path)

    # Run  optimization
    best_params, best_value = run_hpo(
        data=data,
        objective_func=objective,
        alpha=args.alpha,
        n_trials=args.n_trials,
        seed=42,
        verbose=args.verbose,
    )
    print(f"Best parameters: {best_params}")
    print(f"Best value: {best_value}")

    # Save the best parameters to a JSON file
    with open("hpo_best_params.json", "w") as f:
        json.dump(best_params, f, indent=4)
        json.dump(f"\nBest value: {best_value}", f, indent=4)
