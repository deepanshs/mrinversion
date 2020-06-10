# -*- coding: utf-8 -*-
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

    lw = 0.3
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
                    linewidth=lw,
                    alpha=0.5,
                )
            )

    angle1 = np.tan(np.pi * np.asarray([0, 0.2, 0.4, 0.6, 0.8]) / 4.0)
    angle2 = np.tan(np.pi * np.asarray([0.8, 0.6, 0.4, 0.2, 0]) / 4.0)
    for ang_ in angle1:
        ax.plot(x, ((x - offset) * ang_) + offset, "k--", alpha=0.5, linewidth=lw)
    for ang_ in angle2:
        ax.plot(((x - offset) * ang_) + offset, x, "k--", alpha=0.5, linewidth=lw)
    ax.plot(x, x, "k", alpha=0.5, linewidth=2 * lw)
    ax.set_xlim(limx)
    ax.set_ylim(limy)


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

    lw = linewidth

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
        x1, y1, dist_, zdir="z", offset=offz_n, cmap=clr, levels=levels, linewidths=lw
    )

    # 2D x-z contour projection
    x1, y1 = np.meshgrid(a, c, indexing="ij")

    dist = f.sum(axis=1)
    if max_2d[1] is None:
        max_2d[1] = dist.max()
    dist_ = dist / max_2d[1]
    ax.contour(
        x1, dist_, y1, zdir="y", offset=offy, cmap=clr, levels=levels, linewidths=lw
    )

    # 1D x-axis projection
    proj_x = dist.sum(axis=1)
    if max_1d[0] is None:
        max_1d[0] = proj_x.max()
    proj_x /= max_1d[0]
    ax.plot(a, sign * 14 * proj_x + offz, offy, zdir="y", c=ck, linewidth=lw)

    # 1D z-axis projection
    proj_z = dist.sum(axis=0)
    if max_1d[2] is None:
        max_1d[2] = proj_z.max()
    proj_z /= max_1d[2]
    ax.plot(-20 * proj_z + offy_n, c, offx, zdir="x", c=ck, linewidth=lw)
    ax.set_xlim(z_lim)

    # 2D y-z contour projection
    x1, y1 = np.meshgrid(b, c, indexing="ij")
    dist = f.sum(axis=0)
    if max_2d[0] is None:
        max_2d[0] = dist.max()
    dist_ = dist / max_2d[0]
    ax.contour(
        dist_, x1, y1, zdir="x", offset=offx, cmap=clr, levels=levels, linewidths=lw
    )

    # 1D y-axis projection
    proj_y = dist.sum(axis=1)
    if max_1d[1] is None:
        max_1d[1] = proj_y.max()
    proj_y /= max_1d[1]
    ax.plot(
        b, sign * 14 * proj_y + offz, offx, zdir="x", c=ck, linewidth=lw, label=label
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

    l_box = lw
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
