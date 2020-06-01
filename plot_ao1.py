import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline


filename = "logs/aa&ao1.log"
with open(filename, "r") as f:
    history = f.readlines()[:]

n = len(history) // 3

Aass = []
Aoss = []

for i in range(3):
    Aas = []
    Aos = []
    for j in range(n):
        line = history[i * n + j].split()
        Aas.append(float(line[-5]))
        Aos.append(float(line[-1]))
    Aass.append(Aas)
    Aoss.append(Aos)

mAoss = np.mean(np.array(Aoss), axis=1)

y = []
y.append(0.76)
y.extend(mAoss)

x = ["base model", "k=2", "k=3", "k=4"]
# mAass = np.mean(np.array(Aass), axis=0)

ind = np.arange(len(y))  # the x locations for the groups
width = 0.3

fig, ax = plt.subplots(figsize=(9, 7))
rects1 = ax.bar(ind, y, width, color="slateblue", edgecolor="black", linewidth=4,)

# Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel("")
# ax.set_xlabel("Model")
# ax.set_title('Accuracy')
ax.set_xticks(ind)
ax.set_xticklabels(
    ("base model", "parity model k=2", "parity model k=3", "parity model k=4")
)
plt.xticks(size=14)
plt.yticks(size=14)
plt.xlabel("Model", fontdict={"size": 16})
plt.ylabel("Overall Accuracy(percent)", fontdict={"size": 16})
# plt.xticks(fontproperties = 'Times New Roman', size = 14)
# ax.legend(loc='upper right',bbox_to_anchor=(1, 1.15))


def autolabel(rects, xpos="center"):
    """
    Attach a text label above each bar in *rects*, displaying its height.

    *xpos* indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    """

    xpos = xpos.lower()  # normalize the case of the parameter
    ha = {"center": "center", "right": "left", "left": "right"}
    offset = {"center": 0.5, "right": 0.57, "left": 0.43}  # x_txt = x + w*off

    for rect in rects:
        height = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() * offset[xpos],
            1.01 * height,
            "{:.2f}".format(height),
            ha=ha[xpos],
            va="bottom",
            fontdict={"fontsize": 14},
        )


autolabel(rects1)
plt.savefig("results/overall_accuracy.jpg")

# plt.figure(figsize=(8, 6))
# # 绘制条形图
# plt.bar(range(len(y)), y, width=0.3)
# # 对应x轴与字符串
# plt.xticks(range(len(y)), x)
# plt.savefig("tmp6.jpg")
# plot Ao with the growth of error rate
# epoches = np.arange(0.01, 0.11, 0.01)
