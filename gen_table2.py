import numpy as np
import os


class ResultSummarizer:
    def __init__(self, root_path):
        all_optim = np.transpose(np.load(os.path.join(root_path, 'all_optim.npy')))
        all_hess_inv = np.sqrt(np.transpose(np.load(os.path.join(root_path, 'all_hess_inv.npy'))))

        all_optim[:2, :] = np.exp(all_optim[:2, :])
        n_sim = all_optim.shape[1]

        var_e = 0.51
        var_g = 1.74
        k = 4.32
        beta_0 = -35.70
        beta_1 = 0.27
        beta_2 = 0.05

        true_values = np.array([var_e, var_g, k, beta_0, beta_1, beta_2])

        # bias
        self.bias = np.mean(all_optim, axis=1) - true_values
        self.mcse_bias = (1/np.sqrt(n_sim))*np.std(all_optim, axis=1)

        # empSE
        emp_se = np.std(all_optim, axis=1)
        self.emp_se = emp_se
        self.mcse_emp_se = emp_se/np.sqrt(2*(n_sim - 1))

        # average mod_se
        all_hess_inv[:2, :] = all_optim[:2, :]*all_hess_inv[:2, :]
        mod_se = np.sqrt(np.mean(all_hess_inv**2, axis=1))
        self.mod_se = mod_se
        self.mcse_mod_se = np.std(all_hess_inv**2, axis=1)/np.sqrt(4*n_sim*mod_se**2)

        # mod_se_error
        self.mod_se_error = 100*(mod_se/emp_se - 1)
        self.mcse_mod_se_error = 100*(mod_se/emp_se)*np.sqrt(np.std(all_hess_inv**2, axis=1)**2/(4*n_sim*mod_se**4) + 1/(2*(n_sim-1)))

        # coverage, monotone transform
        low = all_optim - 1.96 * all_hess_inv
        up = all_optim + 1.96 * all_hess_inv

        all_optim = np.transpose(np.load(os.path.join(root_path, 'all_optim.npy')))
        all_hess_inv = np.sqrt(np.transpose(np.load(os.path.join(root_path, 'all_hess_inv.npy'))))

        low[:2] = np.exp(all_optim[:2] - 1.96*all_hess_inv[:2])
        up[:2] = np.exp(all_optim[:2] + 1.96*all_hess_inv[:2])

        coverage = np.mean((true_values[:, None] > low) & (true_values[:, None] < up), axis=1)
        self.mcse_coverage = np.sqrt(coverage*(1-coverage)/n_sim)
        self.coverage = coverage


def results_section_str(title, values, mcses, fixed=False):
    nmod = len(values)
    res = title + ' ' + ' '.join(['&']*nmod*2)
    parameter_names = (
        '$\\sigma^2_E$',
        '$\\sigma^2_G$',
        '$k$',
        '$\\beta_0$',
        '$\\beta_1$',
        '$\\beta_2$',
    )

    for i, name in enumerate(parameter_names):
        res += ' \\\\\n\\hspace{2mm} ' + f'{name} & '

        if fixed:
            res += ' & '.join(f'{values[j][i]:.2f} & ({mcses[j][i]:.2f})' for j in range(nmod))
        else:
            res += ' & '.join(f'{values[j][i]:.2e} & ({mcses[j][i]:.2e})' for j in range(nmod))

    return res



if __name__ == '__main__':
    aggf5_results = ResultSummarizer('sim-output/aggf')
    laggf5_results = ResultSummarizer('sim-output/results5')
    laggf18_results = ResultSummarizer('sim-output/results18')
    laggf18_reordered_results = ResultSummarizer('sim-output/results-reordered')
    laggf18_shuffled_results = ResultSummarizer('sim-output/results-shuffled')

    model_results = (
        aggf5_results,
        laggf5_results,
        laggf18_results,
        laggf18_reordered_results,
        laggf18_shuffled_results
    )

    print(results_section_str('Bias', [r.bias for r in model_results], [r.mcse_bias for r in model_results]))
    print('\\\\')
    print(results_section_str('EmpSE', [r.emp_se for r in model_results], [r.mcse_emp_se for r in model_results]))
    print('\\\\')
    print(results_section_str('ModSE', [r.mod_se for r in model_results], [r.mcse_mod_se for r in model_results]))
    print('\\\\')
    print(results_section_str('Relative error in ModSE (\\%)', [r.mod_se_error for r in model_results], [r.mcse_mod_se_error for r in model_results]))
    print('\\\\')
    print(results_section_str('Coverage', [r.coverage for r in model_results], [r.mcse_coverage for r in model_results], fixed=True))
