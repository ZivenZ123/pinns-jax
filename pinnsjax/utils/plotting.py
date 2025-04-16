import os
import logging
from itertools import combinations, product

import matplotlib as mpl
import matplotlib.gridspec as gridspec
import numpy as np
import scipy.io
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata

log = logging.getLogger(__name__)

# # 设置中文字体
# mpl.rcParams['font.sans-serif'] = [
#     'PingFang SC', 'Hiragino Sans GB', 'STHeiti',
#     'SimHei', 'Microsoft YaHei', 'Arial Unicode MS'
# ]
# mpl.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def figsize(scale, nplots=1):
    """根据给定的缩放比例和图表数量计算图表大小。

    :param scale: 图表大小的缩放因子。
    :param nplots: 图表中子图的数量(默认为1)。
    :return: 计算得到的图表大小（英寸）。
    """

    fig_width_pt = 390.0  # 通过LaTeX的\the\textwidth获得
    inches_per_pt = 1.0 / 72.27  # 将pt转换为英寸
    golden_mean = (np.sqrt(5.0) - 1.0) / 2.0  # 美学比例（可以更改）
    fig_width = fig_width_pt * inches_per_pt * scale  # 宽度（英寸）
    fig_height = nplots * fig_width * golden_mean  # 高度（英寸）
    fig_size = [fig_width, fig_height]
    return fig_size


mpl.rcParams.update(mpl.rcParamsDefault)

import matplotlib.pyplot as plt


def newfig(width, nplots=1):
    """创建一个具有指定宽度和子图数量的新图表。

    :param width: 图表的宽度。
    :param nplots: 图表中子图的数量(默认为1)。
    :return: 创建的图表和子图坐标轴。
    """

    fig = plt.figure(figsize=figsize(width, nplots))
    ax = fig.add_subplot(111)
    return fig, ax


def savefig(filename, crop=True):
    """将图表保存到指定文件名，可选择裁剪边缘。

    :param filename: 输出文件的名称（不含扩展名）。
    :param crop: 是否对保存的图像应用紧裁剪(默认为True)。
    """

    log.info(f"图像保存于 {filename}")

    dir_name = os.path.dirname(filename)
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)

    # 设置保存时的字体
    plt.rcParams['font.sans-serif'] = [
        'PingFang SC', 'Hiragino Sans GB', 'STHeiti'
    ]
    plt.rcParams['axes.unicode_minus'] = False

    if crop:
        plt.savefig(f"{filename}.pdf", bbox_inches="tight", pad_inches=0)
        plt.savefig(f"{filename}.eps", bbox_inches="tight", pad_inches=0)
    else:
        plt.savefig(f"{filename}.pdf")
        plt.savefig(f"{filename}.eps")


def plot_navier_stokes(mesh, preds, train_datasets, val_dataset, file_name):
    """绘制纳维-斯托克斯连续反向PDE。"""

    x, t, u = train_datasets[0][:]
    p_star = mesh.solution["p"][:, 100]
    p_pred = preds["p"].reshape(p_star.shape)
    X_star = np.hstack(mesh.spatial_domain)
    lb = X_star.min(0)
    ub = X_star.max(0)
    nn = 200
    x = np.linspace(lb[0], ub[0], nn)
    y = np.linspace(lb[1], ub[1], nn)
    X, Y = np.meshgrid(x, y)
    x_star = X_star[:, 0:1]
    y_star = X_star[:, 1:2]

    PP_star = griddata(X_star, p_pred.flatten(), (X, Y), method="cubic")
    P_exact = griddata(X_star, p_star.flatten(), (X, Y), method="cubic")

    fig, ax = newfig(1.015, 0.8)
    ax.axis("off")
    gs2 = gridspec.GridSpec(1, 2)
    gs2.update(top=1, bottom=1 - 1 / 2, left=0.1, right=0.9, wspace=0.5)
    ax = plt.subplot(gs2[:, 0])
    h = ax.imshow(
        PP_star,
        interpolation="nearest",
        cmap="rainbow",
        extent=[x_star.min(), x_star.max(), y_star.min(), y_star.max()],
        origin="lower",
        aspect="auto",
    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    fig.colorbar(h, cax=cax)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_aspect("equal", "box")
    ax.set_title("Predicted Pressure", fontsize=10)

    # 精确 p(t,x,y)
    ax = plt.subplot(gs2[:, 1])
    h = ax.imshow(
        P_exact,
        interpolation="nearest",
        cmap="rainbow",
        extent=[x_star.min(), x_star.max(), y_star.min(), y_star.max()],
        origin="lower",
        aspect="auto",
    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    fig.colorbar(h, cax=cax)
    ax.set_xlabel("$x$")
    ax.set_ylabel("$y$")
    ax.set_aspect("equal", "box")
    ax.set_title("Exact Pressure", fontsize=10)
    savefig(file_name + "/fig")


def plot_kdv(mesh, preds, train_datasets, val_dataset, file_name):
    """绘制KdV离散反向PDE。"""

    fig, ax = newfig(1.0, 1.2)
    ax.axis("off")

    x0 = train_datasets[0].spatial_domain_sampled[0]
    u0 = train_datasets[0].solution_sampled[0]
    idx_t0 = train_datasets[0].idx_t
    x1 = train_datasets[1].spatial_domain_sampled[0]
    u1 = train_datasets[1].solution_sampled[0]
    idx_t1 = train_datasets[1].idx_t
    exact_u = mesh.solution["u"]

    # 第0行：h(t,x)
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(
        top=1 - 0.06,
        bottom=1 - 1 / 2 + 0.1,
        left=0.15,
        right=0.85,
        wspace=0
    )
    ax = plt.subplot(gs0[:, :])

    h = ax.imshow(
        exact_u,
        interpolation="nearest",
        cmap="rainbow",
        extent=[
            mesh.time_domain[:].min(),
            mesh.time_domain[:].max(),
            mesh.spatial_domain[:].min(),
            mesh.spatial_domain[:].max(),
        ],
        origin="lower",
        aspect="auto",
    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)

    line = np.linspace(
        mesh.spatial_domain[:].min(),
        mesh.spatial_domain[:].max(),
        2
    )[:, None]
    ax.plot(
        mesh.time_domain[idx_t0] * np.ones((2, 1)),
        line,
        "w-",
        linewidth=1
    )
    ax.plot(
        mesh.time_domain[idx_t1] * np.ones((2, 1)),
        line,
        "w-",
        linewidth=1
    )

    ax.set_xlabel("$t$")
    ax.set_ylabel("$x$")
    ax.legend(frameon=False, loc="best")
    ax.set_title("$u(t,x)$", fontsize=10)

    # 第1行：h(t,x) 切片
    gs1 = gridspec.GridSpec(1, 2)
    gs1.update(
        top=1 - 1 / 2 - 0.05,
        bottom=0.15,
        left=0.15,
        right=0.85,
        wspace=0.5
    )

    ax = plt.subplot(gs1[0, 0])
    ax.plot(mesh.spatial_domain[:], exact_u[:, idx_t0].T, "b-", linewidth=2)
    ax.plot(x0, u0, "rx", linewidth=2, label="Data")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$u(t,x)$")
    ax.set_title("$t = %.2f$" % (mesh.time_domain[idx_t0]), fontsize=10)
    ax.set_xlim([mesh.lb[:-1] - 0.1, mesh.ub[:-1] + 0.1])
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.8, -0.3),
        ncol=2,
        frameon=False
    )

    ax = plt.subplot(gs1[0, 1])
    ax.plot(mesh.spatial_domain[:], exact_u[:, idx_t1], "b-", linewidth=2)
    ax.plot(x1, u1, "rx", linewidth=2, label="Data")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$u(t,x)$")
    ax.set_title("$t = %.2f$" % (mesh.time_domain[idx_t1]), fontsize=10)
    ax.set_xlim([mesh.lb[:-1] - 0.1, mesh.ub[:-1] + 0.1])

    savefig(file_name + "/fig")


def plot_ac(mesh, preds, train_datasets, val_dataset, file_name):
    """绘制Allen-Cahn离散前向PDE。"""

    fig, ax = newfig(1.0, 1.2)

    x0 = train_datasets[0].spatial_domain_sampled[0]
    u0 = train_datasets[0].solution_sampled[0]
    exact_u = mesh.solution["u"]
    idx_t0 = train_datasets[0].idx_t
    idx_t1 = val_dataset.idx_t
    U1_pred = preds["u"]

    ax.axis("off")

    # 第0行：h(t,x)
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(
        top=1 - 0.06,
        bottom=1 - 1 / 2 + 0.1,
        left=0.15,
        right=0.85,
        wspace=0
    )
    ax = plt.subplot(gs0[:, :])

    h = ax.imshow(
        exact_u,
        interpolation="nearest",
        cmap="seismic",
        extent=[
            mesh.time_domain[:].min(),
            mesh.time_domain[:].max(),
            mesh.spatial_domain[:].min(),
            mesh.spatial_domain[:].max(),
        ],
        origin="lower",
        aspect="auto",
    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)

    line = np.linspace(
        mesh.spatial_domain[:].min(),
        mesh.spatial_domain[:].max(),
        2
    )[:, None]
    ax.plot(
        mesh.time_domain[idx_t0] * np.ones((2, 1)),
        line,
        "w-",
        linewidth=1
    )
    ax.plot(
        mesh.time_domain[idx_t1] * np.ones((2, 1)),
        line,
        "w-",
        linewidth=1
    )

    ax.set_xlabel("$t$")
    ax.set_ylabel("$x$")
    leg = ax.legend(frameon=False, loc="best")
    ax.set_title("$u(t,x)$", fontsize=10)

    # 第1行：h(t,x) 切片
    gs1 = gridspec.GridSpec(1, 2)
    gs1.update(
        top=1 - 1 / 2 - 0.05,
        bottom=0.15,
        left=0.15,
        right=0.85,
        wspace=0.5
    )

    ax = plt.subplot(gs1[0, 0])
    ax.plot(mesh.spatial_domain[:], exact_u[:, idx_t0], "b-", linewidth=2)
    ax.plot(x0, u0, "rx", linewidth=2, label="Data")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$u(t,x)$")
    ax.set_title("$t = %.2f$" % (mesh.time_domain[idx_t0]), fontsize=10)
    ax.set_xlim([mesh.lb[:-1] - 0.1, mesh.ub[:-1] + 0.1])
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.8, -0.3),
        ncol=2,
        frameon=False
    )

    ax = plt.subplot(gs1[0, 1])
    ax.plot(
        mesh.spatial_domain[:],
        exact_u[:, idx_t1],
        "b-",
        linewidth=2,
        label="Exact Solution"
    )
    ax.plot(
        mesh.spatial_domain[:],
        U1_pred[:, -1],
        "r--",
        linewidth=2,
        label="Prediction"
    )
    ax.set_xlabel("$x$")
    ax.set_ylabel("$u(t,x)$")
    ax.set_title("$t = %.2f$" % (mesh.time_domain[idx_t1]), fontsize=10)
    ax.set_xlim([mesh.lb[:-1] - 0.1, mesh.ub[:-1] + 0.1])

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.1, -0.3),
        ncol=2,
        frameon=False
    )

    savefig(file_name + "/fig")


def plot_burgers_discrete_forward(
    mesh, preds, train_datasets, val_dataset, file_name
):
    """绘制Burgers离散前向PDE。"""

    fig, ax = newfig(1.0, 1.2)

    x0 = train_datasets[0].spatial_domain_sampled[0]
    u0 = train_datasets[0].solution_sampled[0]
    exact_u = mesh.solution["u"]
    idx_t0 = train_datasets[0].idx_t
    idx_t1 = val_dataset.idx_t
    U1_pred = preds["u"]

    ax.axis("off")

    # 第0行：h(t,x)
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(
        top=1 - 0.06,
        bottom=1 - 1 / 2 + 0.1,
        left=0.15,
        right=0.85,
        wspace=0
    )
    ax = plt.subplot(gs0[:, :])

    h = ax.imshow(
        exact_u,
        interpolation="nearest",
        cmap="seismic",
        extent=[
            mesh.time_domain[:].min(),
            mesh.time_domain[:].max(),
            mesh.spatial_domain[:].min(),
            mesh.spatial_domain[:].max(),
        ],
        origin="lower",
        aspect="auto",
    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)

    line = np.linspace(
        mesh.spatial_domain[:].min(),
        mesh.spatial_domain[:].max(),
        2
    )[:, None]
    ax.plot(
        mesh.time_domain[idx_t0] * np.ones((2, 1)),
        line,
        "w-",
        linewidth=1
    )
    ax.plot(
        mesh.time_domain[idx_t1] * np.ones((2, 1)),
        line,
        "w-",
        linewidth=1
    )

    ax.set_xlabel("$t$")
    ax.set_ylabel("$x$")
    leg = ax.legend(frameon=False, loc="best")
    ax.set_title("$u(t,x)$", fontsize=10)

    # 第1行：h(t,x) 切片
    gs1 = gridspec.GridSpec(1, 2)
    gs1.update(
        top=1 - 1 / 2 - 0.05,
        bottom=0.15,
        left=0.15,
        right=0.85,
        wspace=0.5
    )

    ax = plt.subplot(gs1[0, 0])
    ax.plot(
        mesh.spatial_domain[:],
        exact_u[:, idx_t0],
        "b-",
        linewidth=2,
        label="Exact Solution"
    )
    ax.plot(
        x0,
        u0,
        "rx",
        linewidth=2,
        label="Data"
    )
    ax.set_xlabel("$x$")
    ax.set_ylabel("$u(t,x)$")
    ax.set_title("$t = %.2f$" % (mesh.time_domain[idx_t0]), fontsize=10)
    ax.set_xlim([mesh.lb[:-1] - 0.1, mesh.ub[:-1] + 0.1])
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.8, -0.3),
        ncol=2,
        frameon=False
    )

    ax = plt.subplot(gs1[0, 1])
    ax.plot(
        mesh.spatial_domain[:],
        exact_u[:, idx_t1],
        "b-",
        linewidth=2,
        label="Exact Solution"
    )
    ax.plot(
        mesh.spatial_domain[:],
        U1_pred[:, -1],
        "r--",
        linewidth=2,
        label="Prediction"
    )
    ax.set_xlabel("$x$")
    ax.set_ylabel("$u(t,x)$")
    ax.set_title("$t = %.2f$" % (mesh.time_domain[idx_t1]), fontsize=10)
    ax.set_xlim([mesh.lb[:-1] - 0.1, mesh.ub[:-1] + 0.1])
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.1, -0.3),
        ncol=2,
        frameon=False
    )

    savefig(file_name + "/fig")


def plot_burgers_discrete_inverse(
    mesh, preds, train_datasets, val_dataset, file_name
):
    """绘制Burgers连续前向PDE。"""

    fig, ax = newfig(1.0, 1.2)

    x0 = train_datasets[0].spatial_domain_sampled[0]
    u0 = train_datasets[0].solution_sampled[0]
    x1 = train_datasets[1].spatial_domain_sampled[0]
    u1 = train_datasets[1].solution_sampled[0]
    exact_u = mesh.solution["u"]
    idx_t0 = train_datasets[0].idx_t
    idx_t1 = val_dataset.idx_t

    ax.axis("off")

    fig, ax = newfig(1.0, 1.5)
    ax.axis("off")

    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(
        top=1 - 0.06,
        bottom=1 - 1 / 3 + 0.05,
        left=0.15,
        right=0.85,
        wspace=0
    )
    ax = plt.subplot(gs0[:, :])

    h = ax.imshow(
        exact_u,
        interpolation="nearest",
        cmap="rainbow",
        extent=[
            mesh.time_domain[:].min(),
            mesh.time_domain[:].max(),
            mesh.spatial_domain[:].min(),
            mesh.spatial_domain[:].max(),
        ],
        origin="lower",
        aspect="auto",
    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)

    line = np.linspace(
        mesh.spatial_domain[:].min(),
        mesh.spatial_domain[:].max(),
        2
    )[:, None]
    ax.plot(
        mesh.time_domain[idx_t0] * np.ones((2, 1)),
        line,
        "w-",
        linewidth=1
    )
    ax.plot(
        mesh.time_domain[idx_t1] * np.ones((2, 1)),
        line,
        "w-",
        linewidth=1
    )
    ax.set_xlabel("$t$")
    ax.set_ylabel("$x$")
    ax.set_title("$u(t,x)$", fontsize=10)

    gs1 = gridspec.GridSpec(1, 2)
    gs1.update(
        top=1 - 1 / 3 - 0.1,
        bottom=1 - 2 / 3,
        left=0.15,
        right=0.85,
        wspace=0.5
    )

    ax = plt.subplot(gs1[0, 0])
    ax.plot(
        mesh.spatial_domain[:],
        exact_u[:, idx_t0].T,
        "b-",
        linewidth=2
    )
    ax.plot(
        x0,
        u0,
        "rx",
        linewidth=2,
        label="Data"
    )
    ax.set_xlabel("$x$")
    ax.set_ylabel("$u(t,x)$")
    ax.set_title(
        "$t = %.2f$\n%d training data" %
        (mesh.time_domain[idx_t0], u0.shape[0]),
        fontsize=10
    )

    ax = plt.subplot(gs1[0, 1])
    ax.plot(
        mesh.spatial_domain[:],
        exact_u[:, idx_t1],
        "b-",
        linewidth=2,
        label="Exact Solution"
    )
    ax.plot(
        x1,
        u1,
        "rx",
        linewidth=2,
        label="Data"
    )
    ax.set_xlabel("$x$")
    ax.set_ylabel("$u(t,x)$")
    ax.set_title(
        "$t = %.2f$\n%d training data" %
        (mesh.time_domain[idx_t1], u1.shape[0]),
        fontsize=10
    )
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.8, -0.3),
        ncol=2,
        frameon=False
    )

    savefig(file_name + "/fig")


def plot_schrodinger(mesh, preds, train_datasets, val_dataset, file_name):
    """绘制薛定谔连续前向PDE。"""

    h_pred = preds["h"]
    Exact_h = mesh.solution["h"]
    H_pred = h_pred.reshape(Exact_h.shape)

    # 第1行：u(t,x) 切片
    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(top=1 - 1 / 3, bottom=0, left=0.1, right=0.9, wspace=0.5)

    '''
    x0, t0, u0 = train_datasets[1][:]
    x_b, t_b, _ = train_datasets[2][:]
    mid = t_b.shape[0] // 2

    X0 = np.hstack((x0[0], t0))
    X_ub = np.hstack((x_b[0][:mid], t_b[:mid]))
    X_lb = np.hstack((x_b[0][mid:], t_b[mid:]))
    X_u_train = np.vstack([X0, X_lb, X_ub])
    '''
    
    fig, ax = newfig(1.0, 0.9)
    ax.axis("off")

    # 第0行：h(t,x)
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1 - 0.06, bottom=1 - 1 / 3, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs0[:, :])

    h = ax.imshow(
        H_pred,
        interpolation="nearest",
        cmap="YlGnBu",
        extent=[mesh.lb[1], mesh.ub[1], mesh.lb[0], mesh.ub[0]],
        origin="lower",
        aspect="auto",
    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)

    '''
    ax.plot(
        X_u_train[:, 1],
        X_u_train[:, 0],
        "kx",
        label="Data (%d points)" % (X_u_train.shape[0]),
        markersize=4,
        clip_on=False,
    )
    '''
    line = np.linspace(
        mesh.spatial_domain[:].min(),
        mesh.spatial_domain[:].max(),
        2
    )[:, None]

    ax.plot(
        mesh.time_domain[75] * np.ones((2, 1)),
        line,
        "k--",
        linewidth=1
    )
    ax.plot(
        mesh.time_domain[100] * np.ones((2, 1)),
        line,
        "k--",
        linewidth=1
    )
    ax.plot(
        mesh.time_domain[125] * np.ones((2, 1)),
        line,
        "k--",
        linewidth=1
    )

    ax.set_xlabel("$t$")
    ax.set_ylabel("$x$")
    leg = ax.legend(frameon=False, loc="best")
    ax.set_title("$|h(t,x)|$", fontsize=10)

    # 第1行：h(t,x) 切片
    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(top=1 - 1 / 3, bottom=0, left=0.1, right=0.9, wspace=0.5)

    ax = plt.subplot(gs1[0, 0])
    ax.plot(
        mesh.spatial_domain[:],
        Exact_h[:, 75],
        "b-",
        linewidth=2,
        label="Exact Solution"
    )
    ax.plot(
        mesh.spatial_domain[:],
        H_pred[:, 75],
        "r--",
        linewidth=2,
        label="Prediction"
    )
    ax.set_xlabel("$x$")
    ax.set_ylabel("$|h(t,x)|$")
    ax.set_title("$t = %.2f$" % (mesh.time_domain[75]), fontsize=10)
    ax.axis("square")
    ax.set_xlim([-5.1, 5.1])
    ax.set_ylim([-0.1, 5.1])

    ax = plt.subplot(gs1[0, 1])
    ax.plot(
        mesh.spatial_domain[:],
        Exact_h[:, 100],
        "b-",
        linewidth=2,
        label="Exact Solution"
    )
    ax.plot(
        mesh.spatial_domain[:],
        H_pred[:, 100],
        "r--",
        linewidth=2,
        label="Prediction"
    )
    ax.set_xlabel("$x$")
    ax.set_ylabel("$|h(t,x)|$")
    ax.axis("square")
    ax.set_xlim([-5.1, 5.1])
    ax.set_ylim([-0.1, 5.1])
    ax.set_title("$t = %.2f$" % (mesh.time_domain[100]), fontsize=10)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.8),
        ncol=5,
        frameon=False
    )

    ax = plt.subplot(gs1[0, 2])
    ax.plot(
        mesh.spatial_domain[:],
        Exact_h[:, 125],
        "b-",
        linewidth=2,
        label="Exact Solution"
    )
    ax.plot(
        mesh.spatial_domain[:],
        H_pred[:, 125],
        "r--",
        linewidth=2,
        label="Prediction"
    )
    ax.set_xlabel("$x$")
    ax.set_ylabel("$|h(t,x)|$")
    ax.axis("square")
    ax.set_xlim([-5.1, 5.1])
    ax.set_ylim([-0.1, 5.1])
    ax.set_title("$t = %.2f$" % (mesh.time_domain[125]), fontsize=10)

    savefig(file_name + "/fig")


def plot_burgers_continuous_forward(
    mesh, preds, train_datasets, val_dataset, file_name
):
    """绘制Burgers连续前向PDE。"""

    U_pred = preds["u"]
    exact_u = mesh.solution["u"]

    # 处理空间域
    if hasattr(mesh, 'spatial_domain'):
        if isinstance(mesh.spatial_domain, list):
            x = mesh.spatial_domain[0][:]
        else:
            x = mesh.spatial_domain[:]
    else:
        # 假设mesh有坐标网格定义
        x = np.linspace(mesh.lb[0], mesh.ub[0], exact_u.shape[0])

    # 处理训练数据
    x_i, t_i, _ = train_datasets[1][:]
    x_b, t_b, _ = train_datasets[2][:]

    U_pred = U_pred.reshape(exact_u.shape)
    X_u_train = np.vstack([x_i[0], x_b[0]])

    X_u_time = np.vstack([t_i, t_b])

    X_u_train = np.hstack([X_u_train, X_u_time])
    fig, ax = newfig(1.5, 0.9)
    ax.axis("off")

    # 第0行：u(t,x)
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1 - 0.06, bottom=1 - 1 / 3, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs0[:, :])

    # 确定时空域边界
    if (hasattr(mesh, 'time_domain') and
        hasattr(mesh.time_domain, '__getitem__')):
        t_min = np.min(mesh.time_domain[:])
        t_max = np.max(mesh.time_domain[:])
    else:
        t_min = mesh.lb[-1]
        t_max = mesh.ub[-1]

    if hasattr(mesh, 'spatial_domain'):
        if isinstance(mesh.spatial_domain, list):
            x_min = np.min(mesh.spatial_domain[0][:])
            x_max = np.max(mesh.spatial_domain[0][:])
        else:
            try:
                x_min = np.min(mesh.spatial_domain[:])
                x_max = np.max(mesh.spatial_domain[:])
            except AttributeError:
                x_min = np.min(x)
                x_max = np.max(x)
    else:
        x_min = mesh.lb[0]
        x_max = mesh.ub[0]

    h = ax.imshow(
        U_pred,
        interpolation="nearest",
        cmap="rainbow",
        extent=[t_min, t_max, x_min, x_max],
        origin="lower",
        aspect="auto",
    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)

    ax.plot(
        X_u_train[:, 1],
        X_u_train[:, 0],
        "kx",
        label="Data (%d points)" % (X_u_train.shape[0]),
        markersize=4,
        clip_on=False,
    )

    # 绘制参考线
    line = np.linspace(x_min, x_max, 2)[:, None]

    # 确保time_domain访问是安全的
    time_points = (
        mesh.time_domain[:]
        if (hasattr(mesh, 'time_domain') and
            hasattr(mesh.time_domain, '__getitem__'))
        else np.linspace(t_min, t_max, 100)
    )

    # 选择合适的时间点
    t_indices = [25, 50, 75]
    if len(time_points) > max(t_indices):
        ax.plot(time_points[25] * np.ones((2, 1)), line, "w-", linewidth=1)
        ax.plot(time_points[50] * np.ones((2, 1)), line, "w-", linewidth=1)
        ax.plot(time_points[75] * np.ones((2, 1)), line, "w-", linewidth=1)
    else:
        # 如果时间点不够，使用相对位置
        points = np.linspace(0, len(time_points)-1, 4)[1:].astype(int)
        for p in points:
            ax.plot(time_points[p] * np.ones((2, 1)), line, "w-", linewidth=1)

    ax.set_xlabel("$t$")
    ax.set_ylabel("$x$")
    ax.legend(frameon=False, loc="best")
    ax.set_title("$u(t,x)$", fontsize=10)

    # 第1行：u(t,x) 切片
    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(top=1 - 1 / 3, bottom=0, left=0.1, right=0.9, wspace=0.5)

    # 为切片图选择合适的时间点
    if len(time_points) > 75:
        t_slices = [25, 50, 75]
        t_titles = ["$t = 0.25$", "$t = 0.50$", "$t = 0.75$"]
    else:
        # 如果时间点不够，使用相对位置
        t_slices = np.linspace(0, len(time_points)-1, 4)[1:].astype(int)
        t_titles = [f"$t = {time_points[i]:.2f}$" for i in t_slices]

    ax = plt.subplot(gs1[0, 0])
    ax.plot(
        x, exact_u[:, t_slices[0]], "b-", linewidth=2, label="精确解"
    )
    ax.plot(
        x, U_pred[:, t_slices[0]], "r--", linewidth=2, label="预测值"
    )
    ax.set_xlabel("$x$")
    ax.set_ylabel("$u(t,x)$")
    ax.set_title(t_titles[0], fontsize=10)
    ax.axis("square")
    ax.set_xlim([x_min-0.1, x_max+0.1])
    ax.set_ylim([np.min(exact_u)-0.1, np.max(exact_u)+0.1])

    ax = plt.subplot(gs1[0, 1])
    ax.plot(
        x, exact_u[:, t_slices[1]], "b-", linewidth=2, label="精确解"
    )
    ax.plot(
        x, U_pred[:, t_slices[1]], "r--", linewidth=2, label="预测值"
    )
    ax.set_xlabel("$x$")
    ax.set_ylabel("$u(t,x)$")
    ax.axis("square")
    ax.set_xlim([x_min-0.1, x_max+0.1])
    ax.set_ylim([np.min(exact_u)-0.1, np.max(exact_u)+0.1])
    ax.set_title(t_titles[1], fontsize=10)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.35),
        ncol=5,
        frameon=False
    )

    ax = plt.subplot(gs1[0, 2])
    ax.plot(
        x, exact_u[:, t_slices[2]], "b-", linewidth=2, label="精确解"
    )
    ax.plot(x, U_pred[:, t_slices[2]], "r--", linewidth=2, label="Prediction")
    ax.set_xlabel("$x$")
    ax.set_ylabel("$u(t,x)$")
    ax.axis("square")
    ax.set_xlim([x_min-0.1, x_max+0.1])
    ax.set_ylim([np.min(exact_u)-0.1, np.max(exact_u)+0.1])
    ax.set_title(t_titles[2], fontsize=10)

    savefig(file_name + "/fig")


def plot_burgers_continuous_inverse(
    mesh, preds, train_datasets, val_dataset, file_name
):
    """绘制Burgers连续反向PDE。"""

    U_pred = preds["u"]

    exact_u = mesh.solution["u"]
    U_pred = U_pred.reshape(exact_u.shape)

    x_i, t_i, _ = train_datasets[0][:]

    X_u_train = np.hstack([x_i[0], t_i])

    fig, ax = newfig(1.0, 0.9)
    ax.axis("off")

    # 第0行：u(t,x)
    gs0 = gridspec.GridSpec(1, 2)
    gs0.update(top=1 - 0.06, bottom=1 - 1 / 3, left=0.15, right=0.85, wspace=0)
    ax = plt.subplot(gs0[:, :])

    h = ax.imshow(
        U_pred,
        interpolation="nearest",
        cmap="rainbow",
        extent=[
            mesh.time_domain[:].min(),
            mesh.time_domain[:].max(),
            mesh.spatial_domain[:].min(),
            mesh.spatial_domain[:].max(),
        ],
        origin="lower",
        aspect="auto",
    )
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    fig.colorbar(h, cax=cax)

    ax.plot(
        X_u_train[:, 1],
        X_u_train[:, 0],
        "kx",
        label="Data (%d points)" % (X_u_train.shape[0]),
        markersize=4,
        clip_on=False,
    )

    line = np.linspace(
        mesh.spatial_domain[:].min(),
        mesh.spatial_domain[:].max(),
        2
    )[:, None]
    ax.plot(
        mesh.time_domain[25] * np.ones((2, 1)),
        line,
        "w-",
        linewidth=1
    )
    ax.plot(
        mesh.time_domain[50] * np.ones((2, 1)),
        line,
        "w-",
        linewidth=1
    )
    ax.plot(
        mesh.time_domain[75] * np.ones((2, 1)),
        line,
        "w-",
        linewidth=1
    )

    ax.set_xlabel("$t$")
    ax.set_ylabel("$x$")
    ax.legend(frameon=False, loc="best")
    ax.set_title("$u(t,x)$", fontsize=10)

    # 第1行：u(t,x) 切片
    gs1 = gridspec.GridSpec(1, 3)
    gs1.update(top=1 - 1 / 3, bottom=0, left=0.1, right=0.9, wspace=0.5)

    ax = plt.subplot(gs1[0, 0])
    ax.plot(
        mesh.spatial_domain[:],
        exact_u[:, 25],
        "b-",
        linewidth=2,
        label="Exact solution"
    )
    ax.plot(
        mesh.spatial_domain[:],
        U_pred[:, 25],
        "r--",
        linewidth=2,
        label="Prediction"
    )
    ax.set_xlabel("$x$")
    ax.set_ylabel("$u(t,x)$")
    ax.set_title("$t = 0.25$", fontsize=10)
    ax.axis("square")
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])

    ax = plt.subplot(gs1[0, 1])
    ax.plot(
        mesh.spatial_domain[:],
        exact_u[:, 50],
        "b-",
        linewidth=2,
        label="Exact solution"
    )
    ax.plot(
        mesh.spatial_domain[:],
        U_pred[:, 50],
        "r--",
        linewidth=2,
        label="Prediction"
    )
    ax.set_xlabel("$x$")
    ax.set_ylabel("$u(t,x)$")
    ax.axis("square")
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_title("$t = 0.50$", fontsize=10)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.35),
        ncol=5,
        frameon=False
    )

    ax = plt.subplot(gs1[0, 2])
    ax.plot(
        mesh.spatial_domain[:],
        exact_u[:, 75],
        "b-",
        linewidth=2,
        label="Exact solution"
    )
    ax.plot(
        mesh.spatial_domain[:],
        U_pred[:, 75],
        "r--",
        linewidth=2,
        label="Prediction"
    )
    ax.set_xlabel("$x$")
    ax.set_ylabel("$u(t,x)$")
    ax.axis("square")
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.set_title("$t = 0.75$", fontsize=10)

    savefig(file_name + "/fig")
