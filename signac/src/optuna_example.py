"""
Optuna example that optimizes a simple quadratic function.

In this example, we optimize a simple quadratic function. We also demonstrate how to continue an
optimization and to use timeouts.

"""
import numpy as np
import optuna


# Define a simple 2-dimensional objective function whose minimum value is -1 when (x, y) = (0, -1).
def objective(trial):
    x1 = trial.suggest_float("x1", -2, 2)
    x2 = trial.suggest_float("x2", -2, 2)

    fact1a = (x1 + x2 + 1) ** 2
    fact1b = 19 - 14 * x1 + 3 * x1 ** 2 - 14 * x2 + 6 * x1 * x2 + 3 * x2 ** 2
    fact1 = 1 + fact1a * fact1b

    fact2a = (2 * x1 - 3 * x2) ** 2
    fact2b = 18 - 32 * x1 + 12 * x1 ** 2 + 48 * x2 - 36 * x1 * x2 + 27 * x2 ** 2
    fact2 = 30 + fact2a * fact2b

    y = fact1 * fact2

    return y


if __name__ == "__main__":
    # Let us minimize the objective function above.
    print("Running 10 trials...")
    study = optuna.create_study()
    study.optimize(objective, n_trials=10)
    print("Best value: {} (params: {})\n".format(study.best_value, study.best_params))

    # We can continue the optimization as follows.
    print("Running 20 additional trials...")
    study.optimize(objective, n_trials=20)
    print("Best value: {} (params: {})\n".format(study.best_value, study.best_params))

    # We can specify the timeout instead of a number of trials.
    print("Running additional trials in 2 seconds...")
    study.optimize(objective, timeout=200.0)
    print("Best value: {} (params: {})\n".format(study.best_value, study.best_params))
