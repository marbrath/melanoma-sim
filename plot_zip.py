import matplotlib.pyplot as plt
import numpy as np
import os


root_path = 'sim-output/aggf5_results'


def read_results(root_path):
    all_optim = np.transpose(np.load(os.path.join(root_path, 'all_optim.npy')))
    all_hess_inv = np.sqrt(np.transpose(np.load(os.path.join(root_path, 'all_hess_inv.npy'))))

    low = all_optim - 1.96*all_hess_inv
    up = all_optim + 1.96*all_hess_inv

    all_optim[:2] = np.exp(all_optim[:2])
    low[:2] = np.exp(low[:2])
    up[:2] = np.exp(up[:2])

    return all_optim, low, up


def plot_zip(mid_points, lower_bounds, upper_bounds, true_value, arg_name):
    index = np.argsort((mid_points - true_value)**2)
    y_values = np.linspace(0, 100, len(index))

    for i, y in enumerate(y_values):
        plt.plot(
            [
                lower_bounds[index[i]],
                upper_bounds[index[i]]
            ],
            [y, y],
            color='C2' if y < 95 else 'C4'
        )

    plt.axvline(true_value, y_values[0], y_values[-1])
    plt.plot(mid_points[index], y_values, '.')
    plt.title(arg_name)


if __name__ == '__main__':
    args = {
        #0: 'var_e',
        #1: 'var_g',
        #2: 'k',
        #3: 'beta_0',
        4: 'beta_1',
        #5: 'beta_2'
    }

    var_e_ = 0.51
    var_g_ = 1.74
    k_ = 4.32
    beta_0_ = -35.70
    beta_1_ = 0.27
    beta_2_ = 0.05
    true_values = [var_e_, var_g_, k_, beta_0_, beta_1_, beta_2_]

    #dataset_name = 'aggf5'
    #estimates, lower_bounds, upper_bounds = read_results('sim-output/aggf5_results')

    #dataset_name = 'l-aggf5'
    #estimates, lower_bounds, upper_bounds = read_results('sim-output/l-aggf5_results')

    dataset_name = 'l-aggf18 reordered'
    estimates, lower_bounds, upper_bounds = read_results('sim-output/l-aggf18_results/reordered_results')

    #dataset_name = 'l-aggf18 shuffled'
    #estimates, lower_bounds, upper_bounds = read_results('sim-output/l-aggf18_results/shuffled_results')

    for arg_idx in args:
        plt.figure()
        plot_zip(
            estimates[arg_idx],
            lower_bounds[arg_idx],
            upper_bounds[arg_idx],
            true_values[arg_idx],
            f'{dataset_name} {args[arg_idx]}'
        )
    plt.show()