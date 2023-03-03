import pickle
import time
import pylab as pl
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from tabulate import tabulate
from matplotlib.animation import FuncAnimation
from functools import partial
import matplotlib
from util import *

matplotlib.interactive(False)

'''
recod = 0,这个是用来控制我用的一个录屏软件，设置成0即可
plot_history = False  这个是为了制作视频，绘制前几次数据，
        因为每次都要改很多参数，这块比较乱，改成True得调试

主要可以改的参数：        
r0     初始位置r坐标，取值范围：n*M (n>2)
          2M是事件视界的位置，M是黑洞的质量，
          这里实际上是GM/c^2，单位是米，因为G=c=1，省略G，c，就剩下M了
        
u_r    初始径向速度,取值范围: [0,1)，1的时候就是光速
u_fai  初始切向速度，取值同上，但要满足: (u_r * u_r + u_fai * u_fai) ** 0.5 < 1

M      黑洞的质量
t_end  计算的总时长，proper time，单位秒
...,steps=int(t_end * 1000000000))
        这个1000000000可以改，比如现在的代码是计算光子球，计算总时长才0.0001，
        而且速度u_fai很大，所以需要把steps设置的很大，一般设置成1~100都可以，
        但如果计算的步长很长，几万几十万秒，也可以改成小数

frame = np.arange(0, len(xs), 50)
        这个50可以改，代表最后的动画每隔50个steps绘制一个点，1就是每个step的数据都绘制

其它的，ax.set_ylim ax.set_xlim , 是x,y 或 r,φ的绘图范围 看情况改吧

2023年2月4日：
重新推导了速度与E、L的公式
r0=4M 圆轨道 u_fai=(2/4)**0.5      u_r=0
r0=3.5M 圆轨道 r_fai=(2/3)**0.5    u_r=0

2023年2月5日
r0=7M 不稳定圆轨道 u_fai 0.45~0.51左右    u_r=0

2023年2月18日
光子球参数：t_end=0.0001 steps=int(t_end * 2000000000))
'''


def rk4n(funcs, initials, steps=200, x0=0, x1=1):
    dx = (x1 - x0) / steps
    x = np.linspace(x0, x1, steps + 1)
    y0 = np.array(initials)
    y = np.array([initials])
    arrive_horizon = False
    arrive_horizon_time = x1
    for xi in tqdm(x[:-1]):
        k1 = funcs(xi, y0)
        ui = y0 + 0.5 * dx * k1
        k2 = funcs(xi + 0.5 * dx, ui)
        ui = y0 + 0.5 * dx * k2
        k3 = funcs(xi + 0.5 * dx, ui)
        ui = y0 + dx * k3
        k4 = funcs(x0 + dx, ui)
        y0 = y0 + dx * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        if y0[0] < 0:  #
            print('break r<0', y[-1] / M, y0 / M)
            y = np.vstack((y, [0, y0[1], y0[2]]))
            break
        elif len(y) > 250 and y0[1] / M > 1:
            print('u_r>1')
            y = np.vstack((y, [0, y0[1], y0[2]]))
            break
        elif not arrive_horizon and y0[0] < 2 * M:
            arrive_horizon_time = round(xi, 7)
            arrive_horizon = True

        y = np.vstack((y, y0))

    table = [
        ["进入黑洞所需时间", "黑洞内观景时间", "总时间", "旋转圈数"],
        [arrive_horizon_time,
         round(xi - arrive_horizon_time, 7),
         round(xi, 7),
         round((y0[2] - initials[2]) / (2 * np.pi), 3)]
    ]
    y = y[1:]
    return x[:y.shape[0]], y, table


Ms = 1.98892 * 10 ** 30  # mass of sun
c = 299792458
G = 6.67 * 10 ** -11
M = G * Ms * 4 * 10 ** 6 / (c * c)
q = lambda r: 1 - 2 * M / r

r0 = 2.00000000001 * M
u_r = 0.8
u_fai = 0
u = (u_r * u_r + u_fai * u_fai) ** 0.5
gamma = (1 - u * u) ** -0.5
Energy = q(r0) ** 0.5 * gamma
dr_dtau_0 = u_r * gamma * q(r0) ** 0.5 * c
L = r0 * u_fai * gamma * c
table = [
    ["u_r", "dr/dtau", "u_fai", "u", "能量", "角动量"],
    [u_r, dr_dtau_0 / c, u_fai, u, Energy, L / c / M]
]


def geodesic_equations(x, ys):
    f = lambda r: M / r ** 2
    r = ys[0]
    y = f(r) / q(r) * (ys[1] ** 2 - Energy ** 2 * c ** 2) + q(r) * L ** 2 / r ** 3
    fai = L / r ** 2
    return np.array([ys[1], y, fai])


t_end = 200
xs, ys, table1 = rk4n(geodesic_equations, [r0, dr_dtau_0, np.pi / 2], x0=0, x1=t_end, steps=int(t_end * 100))
print(xs.shape, ys.shape)
print(ys[-5:, 1] / M)
frame = np.arange(0, len(xs), 50)
frame = np.hstack((frame, [len(xs) - 1]))

print(tabulate(table))
print(tabulate(table1))
recod = 0
plot_history = False
if plot_history:
    with open('data', 'rb') as f:
        x_history, y_history = pickle.load(f)
    with open('data1', 'rb') as f:
        x_history1, y_history1 = pickle.load(f)
    with open('data2', 'rb') as f:
        x_history2, y_history2 = pickle.load(f)
    with open('data3', 'rb') as f:
        x_history3, y_history3 = pickle.load(f)

font2 = {'family': 'serif', 'color': 'darkred', 'size': 13}
# plot 计算结果
fig3 = plt.figure('raw output')
plt.subplot(3, 1, 1)
plt.plot(xs, ys[:, 0] / M, 'r')
plt.ylim([-0.5, ys[:, 0].max() / M + 1])
plt.ylabel('r/M', fontdict=font2)

plt.subplot(3, 1, 2, )
plt.plot(xs, ys[:, 1] / M, 'g')
plt.ylabel('u_r / M', fontdict=font2)  # dr/dτ

plt.subplot(3, 1, 3)
plt.plot(xs, ys[:, 2], 'b')
plt.ylabel('ϕ', fontdict=font2)
plt.xlabel('τ', fontdict=font2)

# animation of dr/dτ with respect to τ in Cartesian coordinate
y_data = ys[:, 0] / M
fig1, ax = plt.subplots()
line1, = ax.plot([], [], 'o')
xdata = []
ydata = []


def animate_dr_dtau_init():
    ax.set_ylim([y_data.min(), y_data.max() * 1.1])
    ax.set_yticks([r0 / M, round(y_data.max(), 1)])
    # ax.set_yticks(np.arange(y_data.min(), y_data.max() + 1, int(y_data.max() - y_data.min() / 5)))
    if plot_history:
        ax.plot(x_history, y_history[:, 0], '#f27405', alpha=0.3)
        ax.plot(x_history1, y_history1[:, 0], '#f27405', alpha=0.3)
        ax.plot(x_history2, y_history2[:, 0], '#f27405', alpha=0.3)
        ax.plot(x_history3, y_history3[:, 0], '#f27405', alpha=0.3)
    else:
        ax.set_xlim(0, xs[-1] + 5)
        ax.set_xticks([0.0, round(xs[-1], 1)])

    ax.set_xlabel('τ', fontdict=font2)
    ax.set_ylabel('r', fontdict=font2)
    return line1,


def animate_dr_dtau_update(frame, ln, x, y):
    x = [xs[frame]]
    y = [y_data[frame]]
    ln.set_data(x, y)
    xdata.append(xs[frame])
    ydata.append(y_data[frame])
    line1, = ax.plot(xdata, ydata, '#f27405')
    return ln, line1


# animation orbits of particle. r and ϕ in polar coordinate
fai = ys[:, 2]
fig2, ax2 = plt.subplots(subplot_kw={'projection': 'polar'})
line2, = ax2.plot([], [], marker='o', )
fai_data = []
r_data = []


def animate_orbits_init():
    ax2.set_xticks([])
    ax2.axis("off")
    ax2.grid(False)
    rad = np.linspace(0, 2 * np.pi, 360)
    r = 2 * np.ones(len(rad))
    ax2.plot(rad, r, "k", linewidth=2)
    ax2.set_ylim([0, 5.2])
    if plot_history:
        ax2.plot(y_history[:, 2] * M, y_history[:, 0], '#1a80d9', alpha=0.3)
        ax2.plot(y_history1[:, 2] * M, y_history1[:, 0], '#1a80d9', alpha=0.3)
        ax2.plot(y_history2[:, 2] * M, y_history2[:, 0], '#1a80d9', alpha=0.3)
        ax2.plot(y_history3[:, 2] * M, y_history3[:, 0], '#1a80d9', alpha=0.3)
    else:
        ax2.set_ylim([-0., ys[:, 0].max() / M * 1.05])
    circle = pl.Circle((0, 0), 2, transform=ax2.transData._b, color="black", alpha=1)
    ax2.add_artist(circle)
    # ax2.set_thetamin(0)
    # ax2.set_thetamax(180)
    return line1,


def animate_orbits_update(frame, ln, x, y):
    x = [fai[frame]]
    y = [y_data[frame]]
    ln.set_data(x, y)
    fai_data.append(fai[frame])
    r_data.append(y_data[frame])
    line2, = ax2.plot(fai_data, r_data, '#1a80d9')
    return ln, line2


move_figure(fig1, 660, 255)
move_figure(fig2, 20, 255)
move_figure(fig3, 1307, 255)

animate_dr_dtau = FuncAnimation(
    fig1, partial(animate_dr_dtau_update, ln=line1, x=[], y=[]),
    frames=frame,
    init_func=animate_dr_dtau_init, blit=True, interval=1, repeat=False)
animate_orbits = FuncAnimation(
    fig2, partial(animate_orbits_update, ln=line2, x=[], y=[]),
    frames=frame,
    init_func=animate_orbits_init, blit=True, interval=1, repeat=False)

if plot_history:
    with open('data4', 'wb') as f:
        pickle.dump([xs, ys / M], f)
if recod:
    recorder()

plt.show()
