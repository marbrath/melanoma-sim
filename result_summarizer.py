import numpy as np
import os


class ResultSummarizer:
    def __init__(self, root_path, skip_first_sigma_e=False):
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

        self.estimates = all_optim
        self.true_values = true_values

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
        self.mcse_mod_se = np.std(all_hess_inv**2, axis=1)/np.sqrt(4*n_sim*mod_se**2)

        if skip_first_sigma_e:
            mod_se[0] = np.sqrt(np.mean(all_hess_inv[0, 1:]**2))
            self.mcse_mod_se[0] = np.std(all_hess_inv[0, 1:] ** 2) / np.sqrt(4 * n_sim * mod_se[0] ** 2)
        self.mod_se = mod_se

        # mod_se_error
        self.mod_se_error = 100*(mod_se/emp_se - 1)
        self.mcse_mod_se_error = 100*(mod_se/emp_se)*np.sqrt(np.std(all_hess_inv**2, axis=1)**2/(4*n_sim*mod_se**4) + 1/(2*(n_sim-1)))

        if skip_first_sigma_e:
            self.mcse_mod_se_error[0] = 100 * (mod_se[0] / emp_se[0]) * np.sqrt(np.std(all_hess_inv[0, 1:]**2) ** 2 / (4 * n_sim * mod_se[0] ** 4) + 1 / (2 * (n_sim - 1)))

        # coverage, monotone transform
        low = all_optim - 1.96 * all_hess_inv
        up = all_optim + 1.96 * all_hess_inv

        all_optim = np.transpose(np.load(os.path.join(root_path, 'all_optim.npy')))
        all_hess_inv = np.sqrt(np.transpose(np.load(os.path.join(root_path, 'all_hess_inv.npy'))))

        self.centiles = np.abs((all_optim - true_values[:, None])/all_hess_inv)
        self.centiles[:2] = np.abs((all_optim[:2] - np.log(true_values[:2, None]))/all_hess_inv[:2])

        low[:2] = np.exp(all_optim[:2] - 1.96*all_hess_inv[:2])
        up[:2] = np.exp(all_optim[:2] + 1.96*all_hess_inv[:2])

        self.low = low
        self.up = up

        coverage = np.mean((true_values[:, None] > low) & (true_values[:, None] < up), axis=1)
        self.mcse_coverage = np.sqrt(coverage*(1-coverage)/n_sim)
        self.coverage = coverage
