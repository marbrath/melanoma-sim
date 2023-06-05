from result_summarizer import ResultSummarizer


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
