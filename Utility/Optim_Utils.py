import numpy as np
from datetime import datetime as dt

def get_sh_params(n, B, eta):
    """
    Returns the params as required by bohb implementation of SH: [config_count for each rung] and [budget for each rung]
    :param eta: reduction factor
    :param n: starting number of configurations
    :param B: total budget for the sh bracket
    """

    def log_eta(x):
        return np.log(x) / np.log(eta)

    halvings = max(1, int(np.ceil(log_eta(n))))
    config_counts = []
    budgets = []
    for i in range(halvings):
        c = int(n / (eta ** i))
        b = B / c
        b = int(b / halvings)
        config_counts.append(c)
        budgets.append(b)
    return config_counts, budgets


def get_iterations_for_bohb(bohb_instance_count, eta=2, min_budget=2, max_budget=None):
    def log_eta(x):
        return np.log(x) / np.log(eta)

    res = []
    s = 0
    B = eta ** bohb_instance_count
    while int(log_eta(B)) - s > log_eta(s):
        s += 1
    s = max(0, s - 1)
    while s >= 0:
        n = eta ** s
        if B / n / max(1, np.ceil(log_eta(n))) >= min_budget:
            if max_budget is None or B / max(1, np.ceil(log_eta(n))) <= max_budget:
                config_counts, budgets = get_sh_params(n, B, eta)
                temp = {'budgets': budgets, 'config_counts': config_counts}
                res.append(temp)
        s -= 1
    return res


def check_terminal_condition(self):
    """
    Checks terminal condition for optimizer instances
    :return: Returns true if no more iterations is to be run
    """
    if self.time_deadline is not None:
        if (dt.now() - self.experiment_start_time).total_seconds() >= (self.time_deadline * 60):
            return True

    if self.target_acc is not None:
        if self.best_overall_acc >= self.target_acc or self.best_acc_in_last_bohb_instance >= self.target_acc:
            return True

    if self.acc_saturation_check:
        if self.best_overall_acc != 0 and \
                (self.best_acc_in_last_bohb_instance - self.best_overall_acc) < self.acc_saturation_delta:
            return True

    return False