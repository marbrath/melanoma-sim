import matplotlib.pyplot as plt
import numpy as np
from result_summarizer import ResultSummarizer


def plot_zip(mid_points, lower_bounds, upper_bounds, true_value, color, label=None):
    index = np.argsort((mid_points - true_value)**2)
    y_values = np.linspace(0, 100, len(index))

    for i, y in enumerate(y_values):
        plt.plot(
            [
                lower_bounds[index[i]],
                upper_bounds[index[i]]
            ],
            [y, y],
            color=color,
            label=label if i == 0 else None
        )


if __name__ == '__main__':
    args = {
        0: 'var_e',
        1: 'var_g',
        2: 'k',
        3: 'beta_0',
        4: 'beta_1',
        5: 'beta_2'
    }

    var_e_ = 0.51
    var_g_ = 1.74
    k_ = 4.32
    beta_0_ = -35.70
    beta_1_ = 0.27
    beta_2_ = 0.05
    true_values = [var_e_, var_g_, k_, beta_0_, beta_1_, beta_2_]

    aggf5_results = ResultSummarizer('sim-output/aggf')
    laggf5_results = ResultSummarizer('sim-output/results5')
    laggf18_results = ResultSummarizer('sim-output/results18')
    laggf18_reordered_results = ResultSummarizer('sim-output/results-reordered')
    laggf18_shuffled_results = ResultSummarizer('sim-output/results-shuffled')

    results = {
        'aggf5': ResultSummarizer('sim-output/aggf'),
        #'laggf5': ResultSummarizer('sim-output/results5'),
        'laggf18': ResultSummarizer('sim-output/results18'),
        #'laggf18-reordered': ResultSummarizer('sim-output/results-reordered'),
        #'laggf18-shuffled': ResultSummarizer('sim-output/results-shuffled')
    }

    for arg_idx in args:
        plt.figure()
        for i, dataset_name in enumerate(results):
            result = results[dataset_name]
            estimates = result.estimates
            lower_bounds = result.low
            upper_bounds = result.up

            plot_zip(
                estimates[arg_idx],
                lower_bounds[arg_idx],
                upper_bounds[arg_idx],
                true_values[arg_idx],
                f'C{i}',
                label=dataset_name
            )
        plt.axvline(true_values[arg_idx], 0, 100)
        plt.title(args[arg_idx])
        plt.legend()
    plt.show()