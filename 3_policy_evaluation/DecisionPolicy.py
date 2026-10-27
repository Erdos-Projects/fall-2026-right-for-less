import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

class DecisionPolicy:
    def __init__(self, predictions, truths, l=1, show=False):
        self.l = l
        scores, costs = predictions
        S, C = truths

        # Align all DataFrames
        self.scores = scores
        self.costs = costs.loc[scores.index, scores.columns]
        self.true_scores = S.loc[scores.index, scores.columns]
        self.true_costs = C.loc[scores.index, scores.columns]

        self.models = scores.columns.to_list()
        self.prompts = scores.index.to_list()

        # Cache NumPy arrays for efficient computations
        self.S = self.scores.to_numpy()
        self.C = self.costs.to_numpy()
        self.TS = self.true_scores.to_numpy()
        self.TC = self.true_costs.to_numpy()

        self.rows = np.arange(len(self.prompts))

        # Compute policy for the initial lambda
        self.selected_models = self.select_models()
        self.policy_performance = self.compute_performance()
        self.policy_cost = self.compute_cost()

        if show:
            self.show_statistics()

    # --------------------------------------------------
    # Input validation
    # --------------------------------------------------

    def validate_input(self, m, x):
        self.validate_model(m)
        self.validate_prompt(x)

    def validate_model(self, m):
        if m not in self.models:
            raise ValueError(f"Model {m} not found.")

    def validate_prompt(self, x):
        if x not in self.prompts:
            raise ValueError(f"Prompt {x} not found.")

    # --------------------------------------------------
    # Individual model/prompt evaluation
    # --------------------------------------------------

    def model_prompt_score(self, m, x):
        self.validate_input(m, x)
        return self.scores.loc[x, m] - self.l * self.costs.loc[x, m]

    def model_scores(self, x):
        self.validate_prompt(x)
        return (
            self.scores.loc[x] - self.l * self.costs.loc[x]
        ).to_dict()

    def select_model_from_prompt(self, x):
        # returns the model with the highest predicted score for x
        ms = self.model_scores(x)
        return max(ms, key=ms.get)

    # --------------------------------------------------
    # Model selection for initial lambda
    # --------------------------------------------------

    def select_models(self):
        # Objective for each prompt-model pair
        objective = self.S - self.l * self.C

        # Best model for each prompt
        best_indices = objective.argmax(axis=1)
        best_models = np.asarray(self.models)[best_indices]

        # Extract corresponding values
        r = self.rows
        c = best_indices

        return pd.DataFrame({
            "best_model": best_models,
            "predicted_score": self.S[r, c],
            "true_score": self.TS[r, c],
            "predicted_cost": self.C[r, c],
            "true_cost": self.TC[r, c]
        }, index=self.scores.index)

    def compute_performance(self):
        return self.selected_models["true_score"].mean()

    def compute_cost(self):
        return self.selected_models["true_cost"].mean()

    # --------------------------------------------------
    # Evaluate a single lambda
    # --------------------------------------------------

    def evaluate_lambda(self, l):
        objective = self.S - l * self.C
        best_indices = objective.argmax(axis=1)

        true_scores = self.TS[self.rows, best_indices]
        true_costs = self.TC[self.rows, best_indices]

        return true_scores.mean(), true_costs.mean()

    # --------------------------------------------------
    # Vectorized evaluation across multiple lambdas
    # --------------------------------------------------

    def evaluate_lambdas(self, l_values, batch_size=20):
        l_values = np.atleast_1d(
            np.asarray(l_values, dtype=float)
        )

        n_lambdas = len(l_values)

        if n_lambdas == 0:
            return pd.DataFrame(
                columns=["performance", "cost"],
                index=pd.Index([], name="lambda")
            )

        if batch_size is None:
            batch_size = n_lambdas

        if batch_size < 1:
            raise ValueError("batch_size must be positive.")

        per = np.empty(n_lambdas)
        costs = np.empty(n_lambdas)

        for start in range(0, n_lambdas, batch_size):
            end = min(start + batch_size, n_lambdas)
            batch = l_values[start:end]

            # Shape: (batch_size, n_prompts, n_models)
            objective = (
                self.S[None, :, :]
                - batch[:, None, None] * self.C[None, :, :]
            )

            # Shape: (batch_size, n_prompts)
            best_indices = objective.argmax(axis=2)

            # Extract actual scores and costs
            true_scores = self.TS[
                self.rows[None, :], best_indices
            ]

            true_costs = self.TC[
                self.rows[None, :], best_indices
            ]

            # Average across prompts
            per[start:end] = true_scores.mean(axis=1)
            costs[start:end] = true_costs.mean(axis=1)

        return pd.DataFrame({
            "lambda": l_values,
            "performance": per,
            "cost": costs
        }).set_index("lambda")

    def select_lambda_from_threshold(self, l_values, threshold, batch_size=20):
        """
            return the lambda corresponding to the cheapest policy that attains 
            the minimum performance threshold.
        """
        results = self.evaluate_lambdas(l_values, batch_size)
        filtered_results = results[results["performance"] >= threshold]
        if len(filtered_results) == 0:
            print("Minimum threshold was not achieved in the data. Returning `None`.")
            return None
        else:
            return filtered_results["cost"].idxmin()

    # --------------------------------------------------
    # Compute statistics
    # --------------------------------------------------

    def show_statistics(self):
        print(f'===========Statistics for l={self.l}===========\n')
        print('How often each model is selected:')
        display((100 * self.selected_models["best_model"].value_counts(normalize=True)).map('{:.2f}%'.format))
        print('===========================================')
        print(f'Performance: {self.policy_performance:.6f}')
        print(f'Cost: ${self.policy_cost:.6f}')
        print('===========================================')
    
    def evaluate_top_tasks(self, Q, num=10):
        tasks, ids = get_top_tasks(Q, num)
        
        # restrict the results to the top tasks
        select_with_task = self.selected_models.loc[ids]
        
        # append the question type to the selected models
        select_with_task = select_with_task.join(Q["eval_name"], how="left")
        
        # calculate some metrics
        select_with_task["score_residual"] = select_with_task["true_score"] - select_with_task["predicted_score"]
        select_with_task["score_error"] = abs(select_with_task["score_residual"])
        select_with_task["cost_norm_error"] = 100 * abs(select_with_task["true_cost"] / select_with_task["predicted_cost"] - 1)
        
        # compute the average over each task
        _ = select_with_task.groupby("eval_name")[["score_residual", "score_error", "cost_norm_error"]].mean()
        top_tasks = tasks.join(_)
        top_tasks.index.name = "Task"
        top_tasks.columns = ["Count", "Score MR", "Score MAE", "Cost MAPE"]
        
        # format the cost error as a percentage
        top_tasks["Cost MAPE"] = top_tasks["Cost MAPE"].map('{:,.2f}%'.format)
        
        return top_tasks

#-----------------------------------------------------
# General helpers
#-----------------------------------------------------
    
def get_top_tasks(Q, num):
    # retrieve the top n occurring tasks
    tasks = Q["eval_name"].value_counts()[:num].to_frame()
    ids = Q.index[Q["eval_name"].map(lambda t: t in tasks.index.to_list())]
    return tasks, ids