
import matplotlib.pyplot as plt
import numpy as np
import pickle

filename=[]

for k in range(1,5):
    filename.append(
        "logs/base_model_history.pck" if k == 1 else f"logs/parity_model_history_ks{k}.pck"
    )

accs=[]
val_accs=[]

for k in range(4):
    with open(filename[k], "rb") as file_pi:
        history=pickle.load(file_pi)
    
    accs.append(history['acc'][-1])
    val_accs.append(history['val_acc'][-1])

# print(accs)
# accs=[0.91, 0.98, 0.92, 0.92]
ind = np.arange(len(accs))  # the x locations for the groups
width = 0.3  # the width of the bars

fig, ax = plt.subplots(figsize=(9,7))
rects1 = ax.bar(ind - width/2, accs, width, 
                color='slateblue', label='Training accuracy',edgecolor='black',linewidth=4)
rects2 = ax.bar(ind + width/2, val_accs, width, 
                color='lightsteelblue', label='Validation accuracy',edgecolor='black',linewidth=4)

# Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_title('Accuracy')
ax.set_xticks(ind)
ax.set_xticklabels(('base model', 'parity model k=2', 'parity model k=3', 'parity model k=4'))
ax.legend(loc='upper right',bbox_to_anchor=(1, 1.15))
plt.xticks(size=14)
plt.yticks( size = 14)
plt.xlabel('Model', fontdict={ 'size'   : 16})
plt.ylabel('Accuracy(percent)',fontdict={'size':16})
plt.legend(prop={'size': 14})


def autolabel(rects, xpos='center'):
    """
    Attach a text label above each bar in *rects*, displaying its height.

    *xpos* indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    """

    xpos = xpos.lower()  # normalize the case of the parameter
    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0.5, 'right': 0.55, 'left': 0.45}  # x_txt = x + w*off

    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()*offset[xpos], 1.01*height,
                '{:.2f}'.format(height), ha=ha[xpos], va='bottom',fontdict={"fontsize":14})


autolabel(rects1, "center")
autolabel(rects2, "center")

# plt.show()

plt.savefig('results/accs.jpg')

