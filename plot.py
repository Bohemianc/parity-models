# import re


# with open("output.txt","r") as f:
#     lines = f.readlines()

# data = []

# for line in lines:
#     print(re.findall(r"(loss.+val_acc:.+)$",line))


import matplotlib.pyplot as plt
import pickle
import sys

k = 2

filename = (
    "logs/base_model_history.pck" if k == 1 else f"logs/parity_model_history_k{k}.pck"
)
acc_path = "results/base_acc.png" if k == 1 else f"results/parity_k{k}_acc.png"
loss_path = "results/base_loss.png" if k == 1 else f"results/parity_k{k}_loss.png"

with open(filename, "rb") as file_pi:
    history = pickle.load(file_pi)

acc = history["acc"]
val_acc = history["val_acc"]
loss = history["loss"]
val_loss = history["val_loss"]
print(
    f"loss: {loss[-1]} acc: {acc[-1]} val_loss: {val_loss[-1]} val_acc: {val_acc[-1]}"
)
epochs = range(len(acc))

plt.plot(epochs, acc, "bo", label="Training accuracy")
plt.plot(epochs, val_acc, "g", label="Validation accuracy")
plt.title("Training and validation accuracy")
plt.legend()
plt.savefig(acc_path)

plt.figure()
plt.plot(epochs, loss, "bo", label="Training Loss")
plt.plot(epochs, val_loss, "g", label="Validation Loss")
plt.title("Training and validation loss")
plt.legend()
plt.savefig(loss_path)
# plt.show()
