import matplotlib.pyplot as plt
import numpy as np
from result_summarizer import ResultSummarizer


def plot_zip(centiles, lower_bounds, upper_bounds, true_value, inlier_color, outlier_color, label=None, ax=None):
    index = np.argsort(centiles)
    y_values = np.linspace(0, 100, len(index))

    for i, y in enumerate(y_values):
        is_inlier = (true_value > lower_bounds[index[i]]) & (true_value < upper_bounds[index[i]])
        (ax if ax is not None else plt).plot(
            [
                lower_bounds[index[i]],
                upper_bounds[index[i]]
            ],
            [y, y],
            color=inlier_color if is_inlier else outlier_color,
            label=label if i == 0 else None
        )


if __name__ == '__main__':
    args = {
        0: '$\\sigma^2_e$',
        1: '$\\sigma^2_z$',
        2: '$k$',
        3: '$\\beta_0$',
        4: '$\\beta_1$',
        5: '$\\beta_2$'
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
        'AGGF': ResultSummarizer('sim-output/aggf'),
        #'laggf5': ResultSummarizer('sim-output/results5'),
        'L-AGGF': ResultSummarizer('sim-output/results18'),
        #'laggf18-reordered': ResultSummarizer('sim-output/results-reordered'),
        #'laggf18-shuffled': ResultSummarizer('sim-output/results-shuffled')
    }

    fig, axs = plt.subplots(len(results), len(args), sharey=True, facecolor='#fafafa')

    for arg_idx in args:
        lo = 1e8
        hi = -1e8

        for dataset_name in results:
            result = results[dataset_name]
            lo = min(lo, result.low[arg_idx].min())
            hi = max(hi, result.up[arg_idx].max())

        for i, dataset_name in enumerate(results):
            result = results[dataset_name]
            centiles = result.centiles
            lower_bounds = result.low
            upper_bounds = result.up
            ax = axs[i][arg_idx]
            ax.set_facecolor('#fafafa')

            plot_zip(
                centiles[arg_idx],
                lower_bounds[arg_idx],
                upper_bounds[arg_idx],
                true_values[arg_idx],
                'C0', 'C1',
                label=dataset_name,
                ax=ax
            )
            ax.axvline(true_values[arg_idx], 0, 100, color='#fafafa')
            ax.set_title(f'{dataset_name} {args[arg_idx]}')
            ax.set_xlim(lo, hi)
        #plt.axvline(true_values[arg_idx], 0, 100)
        #plt.title(args[arg_idx])
        #plt.legend()
    fig, axs = plt.subplots(1, len(args), facecolor='#fafafa')

    for arg_idx in args:
        lo = 1e8
        hi = -1e8

        for dataset_name in results:
            result = results[dataset_name]
            lo = min(lo, result.low[arg_idx].min())
            hi = max(hi, result.up[arg_idx].max())

        for i, dataset_name in enumerate(results):
            result = results[dataset_name]
            estimates = result.estimates
            lower_bounds = result.low
            upper_bounds = result.up
            centiles = result.centiles
            ax = axs[arg_idx]
            ax.set_facecolor('#fafafa')

            plot_zip(
                centiles[arg_idx],
                lower_bounds[arg_idx],
                upper_bounds[arg_idx],
                true_values[arg_idx],
                f'C{i}',
                f'C{i}',
                label=dataset_name,
                ax=ax
            )
            ax.axvline(true_values[arg_idx], 0, 100, color='w')
            ax.set_title(f'{args[arg_idx]}')
            ax.set_xlim(lo, hi)
    plt.show()