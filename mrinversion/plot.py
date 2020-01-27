# import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

# from matplotlib import gridspec


def get_polar_grids(x, ax, offset):
    l = 0.3
    x = np.asarray(x)
    t1 = plt.Polygon([[0, 0], [0, x[-1]], [x[-1], x[-1]]], color="r", alpha=0.05)
    t2 = plt.Polygon([[0, 0], [x[-1], 0], [x[-1], x[-1]]], color="b", alpha=0.05)

    ax.add_artist(t1)
    ax.add_artist(t2)
    for x_ in x:
        if x_ - offset > 0:
            ax.add_artist(
                plt.Circle(
                    (0, 0),
                    x_ - offset,
                    fill=False,
                    color="k",
                    linestyle="--",
                    linewidth=l,
                    alpha=0.5,
                )
            )

    angle1 = np.tan(np.pi * np.asarray([0, 0.2, 0.4, 0.6, 0.8]) / 4.0)
    angle2 = np.tan(np.pi * np.asarray([0.8, 0.6, 0.4, 0.2, 0]) / 4.0)
    for ang_ in angle1:
        ax.plot(x, ((x - offset) * ang_) + offset, "k--", alpha=0.5, linewidth=l)
    for ang_ in angle2:
        ax.plot(((x - offset) * ang_) + offset, x, "k--", alpha=0.5, linewidth=l)
    ax.plot(x, x, "k", alpha=0.5, linewidth=2 * l)


# from IPython.display import display, clear_output

# import mpld3
# from mpld3 import plugins


# def plot(object, solver=None):

#     colormap = 'coolwarm'
#     with object.out1:

#         signal = object.signal
#         dimensions = object.dimensions
#         extent = object.extent
#         reverse = object.reverse
#         baseUnit = object.baseUnit
#         xViewRange = object.xViewRange
#         yViewRange = object.yViewRange


#         fig, ax = plt.subplots(2, 4, figsize=(15, 6))
#         gs = gridspec.GridSpec(2, 4, width_ratios=[10, 1, 10, 1],
#                   height_ratios=[1, 10])

#         ax0 = plt.subplot(gs[1,0])
#         ax0.set_title('Data')
#         ax1 = plt.subplot(gs[0,0], sharex=ax0)
#         ax1.set_axis_off()
#         ax2 = plt.subplot(gs[1,1], sharey=ax0)
#         ax2.set_axis_off()

#         ax0r = plt.subplot(gs[1,2], sharex=ax0, sharey=ax0)
#         ax0r.set_title('Residue')
#         ax1r = plt.subplot(gs[0,2], sharex=ax0r)
#         ax1r.set_axis_off()
#         ax2r = plt.subplot(gs[1,3], sharey=ax0r)
#         ax2r.set_axis_off()

# #         object.fig, object.ax = plt.subplots(1,1, figsize=(5,3.5))
#         if len(dimensions) == 1:
#             ax0.plot(dimensions[0], np.squeeze(signal.real), 'k')
#             ax0.set_xlim(extent[0], extent[1])
#             ax0.set_xlabel('dimension - 0 / '+ baseUnit[0])
#             if reverse[0]:
#                 ax0.invert_xaxis()
#             if solver is not None:
#                 tempdata = np.squeeze(signal.real.ravel())-
#                           np.squeeze(solver.reconstructedFit)
#             else:
#                 tempdata = np.squeeze(signal.real)
#             ax0r.plot(dimensions[0], tempdata, 'k')
#             ax0r.set_xlim(extent[0], extent[1])
#             ax0r.set_xlabel('inverse dimension / '+ baseUnit[0])
#             if reverse[0]:
#                 ax0r.invert_xaxis()
#     #plt.tight_layout()

#         if len(dimensions) == 2:
#             dimensions[0] = dimensions[0]
#             dimensions[1] = dimensions[1]
#             max = signal.real.max()
#             min = -max

#             proj0 = signal.real.sum(axis=0)
#             proj0Max = proj0.max()
#             proj0Min = proj0.min()

#             proj1 = signal.real.sum(axis=1)
#             proj1Max = proj1.max()
#             proj1Min = proj1.min()

#             for i in range(4):
#                 extent[i] = extent[i]

#     #object.fig = plt.figure(figsize=(8, 6))


#             ax0.imshow(signal.real, extent=(extent[0], extent[1], \
#                             extent[2], extent[3]), vmax=max, vmin=min,\
#                            origin='lower', aspect='auto', cmap=colormap, \
#                            interpolation='none')
#             ax0.set_xlim(xViewRange); ax0.set_ylim(yViewRange)
#             ax0.set_xlabel('dimension - 0 / '+ baseUnit[0])
#             ax0.set_ylabel('dimension - 1 / '+ baseUnit[1])

#             ax1.plot(dimensions[0], proj0, 'k', linewidth=2)
#             ax1.set_ylim([proj0Min, proj0Max])
#             ax2.plot(proj1, dimensions[1], 'k', linewidth=2)
#             ax2.set_xlim([proj1Min, proj1Max])
#             if reverse[0]:
#                ax0.invert_xaxis()
#                ax1.invert_xaxis()
#             if reverse[1]:
#                ax0.invert_yaxis()
#                ax2.invert_yaxis()
#             ax0.grid()

#             if solver is not None:
#                 tempdata = signal.real-solver.reconstructedFit
#             else:
#                 tempdata = signal.real
#             ax0r.imshow(tempdata, extent=(extent[0], extent[1], \
#                            extent[2], extent[3]), vmax=tempdata.max(),
#                            vmin=-tempdata.max(),\
#                            origin='lower', aspect='auto', cmap=colormap, \
#                            interpolation='none')
#             ax0r.set_xlim(xViewRange); ax0.set_ylim(yViewRange)
#             ax0r.set_xlabel('dimension - 0 / '+ baseUnit[0])
#             ax0r.set_ylabel('dimension - 1 / '+ baseUnit[1])

#             proj0 = tempdata.sum(axis=0).real
#             proj0Min = proj0.min()
#             proj0Max = proj0.max()

#             proj1 = tempdata.sum(axis=1).real
#             proj1Min = proj1.min()
#             proj1Max = proj1.max()

#             ax1r.plot(dimensions[0], proj0, 'k', linewidth=2)
#             ax1r.set_ylim([proj0Min, proj0Max])
#             ax2r.plot(proj1, dimensions[1], 'k', linewidth=2)
#             ax2r.set_xlim([proj1Min, proj1Max])
#             if reverse[0]:
#                ax0r.invert_xaxis()
#                ax1r.invert_xaxis()
#             if reverse[1]:
#                ax0r.invert_yaxis()
#                ax2r.invert_yaxis()

#             ax0r.grid()
#                 #plt.subplots_adjust(left=0.125, bottom=0.125,
#                 #    right=0.95 , top=0.95,
#                 #    wspace=0.0, hspace=0.0)
#         clear_output(wait=True)
#         plt.tight_layout()
#         plt.show()
# plugins.connect(fig, plugins.MousePosition(fontsize=14))
# display(mpld3.display())


def plotProjected(
    signal,
    projectedSignal,
    dimensions,
    extent,
    reverse,
    baseUnit,
    xViewRange,
    yViewRange=None,
):
    if len(dimensions) == 1:
        dimensions[0] = dimensions[0]
        plt.plot(dimensions[0], signal.real, "k")
        plt.plot(dimensions[0], projectedSignal.real, "r")
        plt.xlim(extent[0], extent[1])
        plt.xlabel("dimension - 0 / " + baseUnit[0])
        if reverse[0]:
            plt.gca().invert_xaxis()

    if len(dimensions) == 2:
        for i in range(4):
            extent[i] = extent[i]
        dimensions[0] = dimensions[0]
        dimensions[1] = dimensions[1]
        extent[i] = extent[i]
        fig, ax = plt.subplots(2, 2)
        max = signal.real.max()
        min = signal.real.min()
        ax[0, 0].imshow(
            signal.real,
            extent=(extent[0].value, extent[1].value, extent[2].value, extent[3].value),
            vmax=max,
            vmin=min,
            origin="lower",
            aspect="auto",
            cmap="coolwarm",
        )
        ax[0, 0].set_xlim(xViewRange)
        ax[0, 0].set_ylim(yViewRange)
        ax[0, 0].set_xlabel("dimension - 0 / " + baseUnit[0])
        ax[0, 0].set_ylabel("dimension - 1 / " + baseUnit[1])
        ax[0, 0].set_title("Data")

        ax[0, 1].imshow(
            projectedSignal.real,
            extent=(extent[0].value, extent[1].value, extent[2].value, extent[3].value),
            vmax=max,
            vmin=min,
            origin="lower",
            aspect="auto",
            cmap="coolwarm",
        )
        ax[0, 1].set_xlim(xViewRange)
        ax[0, 1].set_ylim(yViewRange)
        ax[0, 1].set_xlabel("dimension - 0 / " + baseUnit[0])
        ax[0, 1].set_ylabel("dimension - 1 / " + baseUnit[1])
        ax[0, 1].set_title("Compressed data")

        ax[1, 0].imshow(
            (signal - projectedSignal).real,
            extent=(extent[0].value, extent[1].value, extent[2].value, extent[3].value),
            vmax=max,
            vmin=min,
            origin="lower",
            aspect="auto",
            cmap="coolwarm",
        )
        ax[1, 0].set_xlim(xViewRange)
        ax[1, 0].set_ylim(yViewRange)
        ax[1, 0].set_xlabel("dimension - 0 / " + baseUnit[0])
        ax[1, 0].set_ylabel("dimension - 1 / " + baseUnit[1])
        ax[1, 0].set_title("Residue")

        if reverse[0]:
            ax[0, 0].invert_xaxis()
            ax[0, 1].invert_xaxis()
            ax[1, 0].invert_xaxis()
        if reverse[1]:
            ax[0, 0].invert_yaxis()
            ax[0, 1].invert_yaxis()
            ax[1, 0].invert_yaxis()

    plt.tight_layout()
    plt.show()
