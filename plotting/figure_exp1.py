import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}",
        "font.family": "Computer Modern Roman",
        "font.size": 14
    })

    fname_out = "../results/experiment1/ecdf_instances.pdf"

    list_n = [5, 6, 7]
    map_n_axes = {5: 0, 6: 1, 7: 2}

    fig, axes = plt.subplots(1, 3)
    fig.set_figheight(4)
    fig.set_figwidth(12)

    linestyles = ['solid', 'dotted', 'dashed', 'dashdot']
    colors = ['blue', 'red', 'green', 'black']
    names_legend = [
        r'$\mathcal{E}_{\rm{s}}$', r'$\mathcal{E}_{\rm{s1}}$',
        r'$\mathcal{E}_{\rm{s2}}$', r'$\mathcal{L}$'
    ]

    left_xlim = 1.0
    right_xlim = 1.5

    for n in list_n:
        filename = f'../results/experiment1/trimmed_flops_n{n}.npz'
        array = np.load(filename)['a']
        ax = axes[map_n_axes[n]]

        for j in range(array.shape[1]):
            sns.ecdfplot(data=array[:, j],
                         color=colors[j],
                         linestyle=linestyles[j],
                         ax=ax,
                         label=names_legend[j])
        ax.set_ylabel("")
        ax.grid()

        ax.set_title(r'$n={0}$'.format(n))
        ax.set_xlim(left_xlim, right_xlim)
        ax.set_xticks([1.0, 1.1, 1.2, 1.3, 1.4, 1.5])
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels([0, 20, 40, 60, 80, 100])

        if n == 5:
            ax.set_ylabel(r'$\text{Percentage of instances }[\%]$')

        if n == 6:
            ax.set_xlabel(r'$\text{Ratio over optimal number of FLOPs}$')

        if n == 7:
            ax.legend(loc="lower right")

    fig.subplots_adjust(left=0.06,
                        right=0.97,
                        bottom=0.13,
                        top=0.92,
                        wspace=0.29)
    plt.show()
    # plt.savefig(fname_out, format='pdf')
    # print("Figure generated at", fname_out)
