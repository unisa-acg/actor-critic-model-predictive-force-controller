import matplotlib.pyplot as plt
import numpy as np


def plot_explored_space_3D(x, y, f, elev=None, azim=None, vmax=None, vmin=None):
    """Plot the info registered vs the info processed.
    After calling get_info_robot() and contact_info_processed_to_csv() method.

    Returns:
        fig
    """

    fig = plt.figure()

    ax = fig.add_subplot(111, projection="3d")
    p = ax.scatter(x,
                   y,
                   f,
                   linewidths=0.0001,
                   alpha=1,
                   c=f,
                   cmap="viridis",
                   vmin=vmin,
                   vmax=vmax)
    ax.view_init(elev=elev, azim=azim, vertical_axis="z")

    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

    ax.ticklabel_format(style="sci", scilimits=(0, 0))

    if elev == 0 and azim == -90:
        ax.set_yticks([])
        ax.set_xlabel("Position [m]")
        ax.set_zlabel("Force [N]")
    elif elev == 0 and azim == 0:
        ax.set_xticks([])
        ax.set_ylabel("Velocity [m/s]")
        ax.set_zlabel("Force [N]")
    elif elev == 90 and azim == -90:
        ax.set_zticks([])
        # divider = make_axes_locatable(ax)
        # print('1',ax.get_position().x1+0.01)
        # print('2',ax.get_position().y0)
        # print('3','0.02')
        # print('4',ax.get_position().height)

        cax = fig.add_axes([
            ax.get_position().x1 - 0.01,
            ax.get_position().y0 + 0.16, 0.02,
            ax.get_position().height - 0.3
        ])

        fig.colorbar(p, cax=cax, label="Force [N]")
        ax.set_xlabel("Position [m]")
        ax.set_ylabel("Velocity [m/s]")
    else:
        ax.set_xlabel("Position [m]")
        ax.set_ylabel("Velocity [m/s]")
        ax.set_zlabel("Force [N]")
    plt.tight_layout()


def distributions_plot(x, bins=250, showfliers=True):

    title_hist = [
        "Position distribution z-axis",
        "Velocity distribution z-axis",
        "Force distribution z-axis",
    ]
    color = ["#37CAEC", "#2A93D5", "#125488"]
    udm = ['[m]', '[m/s]', '[N]']
    _, axs = plt.subplots(2, 3, gridspec_kw={"height_ratios": [2, 1]})

    for i in range(3):

        axs[0, i].hist(x[i], bins=bins, color=color[i], edgecolor="white")
        axs[0, i].set_title(title_hist[i])
        axs[0, i].ticklabel_format(style="sci", scilimits=(0, 0))
        axs[0, i].set_axisbelow(True)
        axs[0, i].yaxis.grid()
        axs[0, i].set_xlabel(udm[i])

        bp = axs[1, i].boxplot(x[i],
                               widths=0.5,
                               patch_artist=True,
                               vert=False,
                               showfliers=showfliers,
                               flierprops={
                                   'marker': 'o',
                                   'markersize': 2
                               })
        for patch in bp["boxes"]:
            patch.set_facecolor(color[i])
        axs[1, i].ticklabel_format(axis='x', style="sci", scilimits=(0, 0))
        axs[1, i].set_yticks([])
        axs[1, i].set_xlabel(udm[i])
    # plt.tight_layout()


def plot_dataset_along_time(pos_z, vel_z, f_z_out):

    _, axs = plt.subplots(3, 1)
    # plt.tight_layout()
    step_vect = np.arange(0, len(pos_z), 1) * 0.002

    axs[0].plot(step_vect, pos_z, color="#37CAEC")
    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("Position [m]")
    axs[0].ticklabel_format(style="sci", scilimits=(0, 0))
    axs[0].grid()

    axs[1].plot(step_vect, vel_z, color="#2A93D5")
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("Velocity [m/s]")
    axs[1].ticklabel_format(style="sci", scilimits=(0, 0))
    axs[1].grid()

    axs[2].plot(step_vect, f_z_out, color="#125488")
    axs[2].set_xlabel("Time")
    axs[2].set_ylabel("Force [N]")
    axs[2].ticklabel_format(style="sci", scilimits=(0, 0))
    axs[2].grid()


def prediction_images(actual, predicted, epochs, lr, hidden_dim, num_hidden_layers,
                      n_estimators, type, now):
    steps = np.arange(0, len(actual))

    fig = plt.figure()
    f_predicted = [item[0] for item in predicted]
    f_actual = [item[0] for item in actual]
    plt.plot(steps, f_predicted, 'b')
    plt.plot(steps, f_actual, 'r')
    plt.xlabel('Steps')
    plt.ylabel('Force [N]')
    plt.legend(['Predicted Force', 'Actual Force'])
    txt = (
        f'epo = {epochs}, lr = {lr}, neur = {hidden_dim}, layers = {num_hidden_layers}, n_estim = {n_estimators}, ens_type = {type}'
    )
    plt.figtext(0.5, 0.01, txt, wrap=True, horizontalalignment='center', fontsize=12)
    plt.grid()
    # fig.savefig(f'images/prediction{now}.png')

    # plt.figure()
    # pos_predicted = [item[1] for item in predicted]
    # pos_actual = [item[1] for item in actual]
    # plt.plot(steps, pos_predicted, 'b')
    # plt.plot(steps, pos_actual, 'r')
    # plt.xlabel('Steps')
    # plt.ylabel('Position [m]')
    # plt.legend(['Predicted Position', 'Actual Position'])
    # plt.grid()
