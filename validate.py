import numpy as np
import matplotlib.pyplot as plt
import datasets
import utils
import os


def multiple_bars(series):
    for i, s in enumerate(series):
        plt.barh(y=np.arange(len(s)) + i / len(series), width=list(s), height=1/len(series), label="TMC")

"""
sh, sh_var, shs = utils.average_heavysims('22-10000-0.001-0.5')

#multiple_bars(shs + [sh])
plt.barh(y=np.arange(len(sh)), width=list(sh))
plt.show()
"""

sh_f, sh_f_var, _ = utils.average_heavysims('22-10000-0.001-0.5')
sh_q, sh_q_var, _ = utils.average_heavysims('125-10000-0.001-0.5')

print(f"sum(sh_f) = {np.sum(sh_f)}")
print(f"sum(sh_q) = {np.sum(sh_q)}")

_, _, fds = datasets.agaricus_lepiota()

L = [fd.num_values for fd in fds]

gs = utils.one_hot_groups(L)
sh_f_i = np.zeros(len(sh_f))
sh_f_i_var = np.zeros(len(sh_f))

for i, (a, b) in enumerate(gs):
    cs = np.sum(sh_q[a : b+1])
    cs_var = np.sum(sh_q_var[a : b+1])
    sh_f_i[i] = cs
    sh_f_i_var[i] = cs_var

plt.barh(y=np.arange(len(sh_f)), width=list(sh_f), xerr=np.sqrt(sh_f_var)*2, height=0.5, label="TMC")
plt.barh(y=np.arange(len(sh_f_i))+0.5, width=list(sh_f_i), xerr=np.sqrt(sh_f_i_var)*2, height=0.5, label="summed")
plt.legend()
plt.show()

quit()

plt.subplot(1, 2, 1)
plt.barh(y=np.arange(len(sh1)), width=list(sh1), height=0.33)
plt.barh(y=np.arange(len(sh2))+0.33, width=list(sh2), height=0.33)
plt.barh(y=np.arange(len(sh3))+0.66, width=list(sh3), height=0.33)
plt.subplot(1, 2, 2)
m = (sh1 + sh2 + sh3) / 3
plt.barh(y=np.arange(len(sh2)), width=list(m))
plt.show()

quit()

sh1 = np.load("500-0.005-0.0.0.npy")
sh2 = np.load("500-0.005-0.5.0.npy")

n = len(sh1)

d = sh1 - sh2
td = 1 / (2 * n)

print(f"E[sh1 - sh2]   = {np.mean(d)}")
print(f"Var[sh1 - sh2] = {np.var(d)}")

print()
print(f"Expected average difference = 1 / 2n = {td}")

plt.subplot(2, 1, 1)
plt.title("Shapley values")
plt.plot(sh1, label="V({}) = 0.0")
plt.plot(sh2, label="V({}) = 0.5")
plt.plot([0, len(sh1)-1], [td, td], '--k')
plt.legend()

plt.subplot(2, 1, 2)
plt.title("Difference")
plt.plot(d)

plt.show()
