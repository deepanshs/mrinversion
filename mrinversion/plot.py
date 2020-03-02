from itertools import combinations
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm


def get_polar_grids(ax, ticks=None, offset=0):
    limy = ax.get_ylim()
    limx = ax.get_xlim()
    if ticks is None:
        x = np.asarray(ax.get_xticks())
        inc = x[1] - x[0]
        size = x.size
        x = np.arange(size + 5) * inc
    else:
        x = np.asarray(ticks)

    l = 0.3
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
    ax.set_xlim(limx)
    ax.set_ylim(limy)


class ThreeDScatter:
    def __init__(self, ax, csdm_object, theta_angle=28, angle=-150):
        self.max_2d = [
            csdm_object.sum(axis=0).max()[0].value,
            csdm_object.sum(axis=1).max()[0].value,
            csdm_object.sum(axis=2).max()[0].value,
        ]

        self.max_1d = [
            csdm_object.sum(axis=(1, 2)).max()[0].value,
            csdm_object.sum(axis=(0, 2)).max()[0].value,
            csdm_object.sum(axis=(0, 1)).max()[0].value,
        ]
        self._objects = []

    def add(self, x):
        self._objects.append(x)
        pass


def plot_3d(
    ax,
    csdm_object,
    theta_angle=28,
    angle=-150,
    x_lim=None,
    y_lim=None,
    z_lim=None,
    max_2d=None,
    max_1d=None,
    cmap=cm.PiYG,
    box=False,
    clip_percent=0.0,
    linewidth=1,
):

    if max_2d is None:
        max_2d = [None, None, None]
    if max_1d is None:
        max_1d = [None, None, None]

    l = linewidth

    f = csdm_object.dependent_variables[0].components[0].T
    label = csdm_object.description

    a_, b_, c_ = [item for item in csdm_object.dimensions]

    a = a_.coordinates.value
    b = b_.coordinates.value
    c = c_.coordinates.value

    clr = cmap
    ck = cmap(0)
    facecolors = cmap(f)
    facecolors[:, :, :, -1] = f / 9.0
    index = np.where(f < clip_percent / 100)
    facecolors[:, :, :, -1][index] = 0
    facecolors.shape = (f.size, 4)

    if x_lim is None:
        x_lim = [a[0], a[-1]]
    if y_lim is None:
        y_lim = [b[0], b[-1]]

    if z_lim is None:
        offz = c[0] + (c[1] - c[0]) / 2
        offz_n = c[-1]
        sign = np.sign(c[0] - c[-1])
    else:
        offz = z_lim[1] - (c[1] - c[0]) / 2
        offz_n = z_lim[0]
        sign = np.sign(z_lim[1] - z_lim[0])

    offy = y_lim[1] + (b[1] - b[0]) / 2
    offy_n = y_lim[0] - (b[1] - b[0]) / 2

    offx = x_lim[1] + (a[1] - a[0]) / 2
    # if angle < -90:
    #     offx = x_lim[1] + (a[1] - a[0]) / 2
    # else:
    #     offx = x_lim[0] - (a[1] - a[0]) / 2

    if angle > 0:
        offy = y_lim[0] - (b[1] - b[0]) / 2
        offy_n = y_lim[-1] + (b[1] - b[0]) / 2
        offx = x_lim[0] - (a[1] - a[0]) / 2

    ax.set_proj_type("persp")
    ax.view_init(theta_angle, angle)

    # 2D x-y contour projection
    x1, y1 = np.meshgrid(a, b, indexing="ij")
    dist = f.sum(axis=2)
    if max_2d[2] is None:
        max_2d[2] = dist.max()
    dist_ = dist / max_2d[2]
    levels = (np.arange(20) + 1) / 20
    ax.contour(
        x1, y1, dist_, zdir="z", offset=offz_n, cmap=clr, levels=levels, linewidths=l
    )

    # 2D x-z contour projection
    x1, y1 = np.meshgrid(a, c, indexing="ij")

    dist = f.sum(axis=1)
    if max_2d[1] is None:
        max_2d[1] = dist.max()
    dist_ = dist / max_2d[1]
    ax.contour(
        x1, dist_, y1, zdir="y", offset=offy, cmap=clr, levels=levels, linewidths=l
    )

    # 1D x-axis projection
    proj_x = dist.sum(axis=1)
    if max_1d[0] is None:
        max_1d[0] = proj_x.max()
    proj_x /= max_1d[0]
    ax.plot(a, sign * 14 * proj_x + offz, offy, zdir="y", c=ck, linewidth=l)

    # 1D z-axis projection
    proj_z = dist.sum(axis=0)
    if max_1d[2] is None:
        max_1d[2] = proj_z.max()
    proj_z /= max_1d[2]
    ax.plot(-20 * proj_z + offy_n, c, offx, zdir="x", c=ck, linewidth=l)
    ax.set_xlim(z_lim)

    # 2D y-z contour projection
    x1, y1 = np.meshgrid(b, c, indexing="ij")
    dist = f.sum(axis=0)
    if max_2d[0] is None:
        max_2d[0] = dist.max()
    dist_ = dist / max_2d[0]
    ax.contour(
        dist_, x1, y1, zdir="x", offset=offx, cmap=clr, levels=levels, linewidths=l
    )

    # 1D y-axis projection
    proj_y = dist.sum(axis=1)
    if max_1d[1] is None:
        max_1d[1] = proj_y.max()
    proj_y /= max_1d[1]
    ax.plot(
        b, sign * 14 * proj_y + offz, offx, zdir="x", c=ck, linewidth=l, label=label
    )

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    if z_lim is None:
        ax.set_zlim([c[-1], c[0]])
    else:
        ax.set_zlim(z_lim)

    ax.set_xlabel(f"{a_.axis_label} - 0")
    ax.set_ylabel(f"{b_.axis_label} - 1")
    ax.set_zlabel(f"{c_.axis_label} - 2")

    x, y, z = np.meshgrid(a, b, c, indexing="ij")
    ax.scatter(x.flat, y.flat, z.flat, marker="X", s=150, c=facecolors)

    # full box
    da = a[1] - a[0]
    r1 = [x_lim[0] - da / 2, x_lim[-1] + da / 2]
    db = b[1] - b[0]
    r2 = [y_lim[0] - db / 2, y_lim[-1] + db / 2]
    dc = c[1] - c[0]
    r3 = [z_lim[-1] - dc / 2, z_lim[0] + dc / 2]

    l_box = l
    for s, e in combinations(np.array(list(product(r1, r2, r3))), 2):
        if np.sum(np.abs(s - e)) == r1[1] - r1[0]:
            ax.plot3D(*zip(s, e), color="gray", linewidth=l_box)
        if np.sum(np.abs(s - e)) == r2[1] - r2[0]:
            ax.plot3D(*zip(s, e), color="gray", linewidth=l_box)
        if np.sum(np.abs(s - e)) == r3[1] - r3[0]:
            ax.plot3D(*zip(s, e), color="gray", linewidth=l_box)

    # draw cube
    if box:
        da = a[1] - a[0]
        r1 = [a[0] - da / 2, a[-1] + da / 2]
        db = b[1] - b[0]
        r2 = [b[0] - db / 2, b[-1] + db / 2]
        dc = c[1] - c[0]
        r3 = [c[0] - dc / 2, c[-1] + dc / 2]

        for s, e in combinations(np.array(list(product(r1, r2, r3))), 2):
            if np.sum(np.abs(s - e)) == r1[1] - r1[0]:
                ax.plot3D(*zip(s, e), c="blue", linestyle="dashed", linewidth=l_box)
            if np.sum(np.abs(s - e)) == r2[1] - r2[0]:
                ax.plot3D(*zip(s, e), c="blue", linestyle="dashed", linewidth=l_box)
            if np.sum(np.abs(s - e)) == r3[1] - r3[0]:
                ax.plot3D(*zip(s, e), c="blue", linestyle="dashed", linewidth=l_box)

    ax.xaxis._axinfo["grid"].update({"linewidth": 0.25, "color": "gray"})
    ax.yaxis._axinfo["grid"].update({"linewidth": 0.25, "color": "gray"})
    ax.zaxis._axinfo["grid"].update({"linewidth": 0.25, "color": "gray"})


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
