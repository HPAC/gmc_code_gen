import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}",
        "font.family": "Computer Modern Roman",
        "font.size": 12
    })

    path_base = '../results/experiment2/ratios_optimal_sets_w'
    arr_flops = np.genfromtxt(path_base + 'flops.txt', dtype=np.float32)
    arr_models = np.genfromtxt(path_base + 'models.txt', dtype=np.float32)

    fig, ax = plt.subplots()
    fig.set_figheight(4)
    fig.set_figwidth(4.5)

    # All ratios are computed wrt the cheapest execution time in \mathcal{A}
    ratios_Es = arr_flops[:, 4]
    ratios_Es1_flops = arr_flops[:, 5]
    ratios_Es1_models = arr_models[:, 5]
    ratios_LtR = arr_models[:, 7]
    ratios_Arma = arr_models[:, 8]

    all_ratios = [
        ratios_Es, ratios_Es1_flops, ratios_Es1_models, ratios_LtR, ratios_Arma
    ]

    legend_entries = [
        r'$\mathcal{E}_{\rm{s}}$', r'$\mathcal{E}_{\rm{s1,F}}$',
        r'$\mathcal{E}_{\rm{s1,M}}$', r'$\mathcal{L}$', r'$\text{Arma}$'
    ]
    colors = ['blue', 'red', 'green', 'black', 'black']
    linestyles = ['solid', 'dotted', 'dashed', 'dashdot', 'dotted']

    for j in range(len(all_ratios)):
        sns.ecdfplot(data=all_ratios[j],
                     color=colors[j],
                     linestyle=linestyles[j],
                     ax=ax,
                     label=legend_entries[j])

    ax.grid()
    ax.legend(loc="lower right")

    # Fix xlims, xticks, yticks, and yticklabels
    ax.set_xlim(1.0, 3.0)
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels([0, 20, 40, 60, 80, 100])

    # Fix ylabel and xlabel
    ax.set_ylabel(r'$\text{Percentage of instances }[\%]$')
    ax.set_xlabel(r'$\text{Ratio over optimal execution time}$')

    # Fix margins figure
    fig.subplots_adjust(left=0.13, right=0.97, bottom=0.12, top=0.95)

    # plt.savefig('figures/exp2/exp2_ratio_optimal.pdf', format='pdf')
    plt.show()
