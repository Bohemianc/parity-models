import numpy as np
import tensorflow as tf
import os
from PIL import Image
from glob import glob
from encoder import encoder

k = 4
img_size = 150
epoches = 500
err_rate = 0.1

base_model = tf.keras.models.load_model(os.path.join("models", "base_model.h5"))
parity_model = tf.keras.models.load_model(
    os.path.join("models", f"parity_model_k{k}.h5")
)

validation_dir = "cats_and_dogs_filtered/validation/"
cats_dir = os.path.join(validation_dir, "cats")
dogs_dir = os.path.join(validation_dir, "dogs")
cats_list = glob(os.path.join(cats_dir, "*.jpg"))
dogs_list = glob(os.path.join(dogs_dir, "*.jpg"))


def get_xs(data_list):
    xs = []
    for file in data_list:
        img = tf.keras.preprocessing.image.load_img(file, target_size=(150, 150))
        x = tf.keras.preprocessing.image.img_to_array(img)
        xs.append(np.expand_dims(x, axis=0))
    return xs


# get true labels of Xs
def get_xs_ys(data_list, cats_num):
    xs_tmp = get_xs(data_list)
    ys_tmp = [0] * cats_num
    ys_tmp.extend([1] * (k - cats_num))

    # ords = np.arange(k)
    ords = [x for x in range(k)]
    xs = []
    ys = []
    for i in ords:
        xs.append(xs_tmp[i])
        ys.append(ys_tmp[i])
    return (xs, ys)


# get ys_hat with the base model
def get_ys_hat(xs):
    ys_hat = []
    for x in xs:
        images = np.vstack([x])
        classes = base_model.predict(images, batch_size=10)
        res = 1 if classes[0] > 0.5 else 0
        ys_hat.append(res)
    return ys_hat


# get the predicition of the parity image with the parity model
def get_fp(parity):
    x = tf.keras.preprocessing.image.img_to_array(parity)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    classes = parity_model.predict(images, batch_size=10)

    maxv = 0
    dogs_num = 0
    for i in range(k + 1):
        if classes[0][i] > maxv:
            maxv = classes[0][i]
            dogs_num = i

    return dogs_num


# get outputs of queries with the base model and the parity model
# and simulate errors
def query(ys_hat, parity_query):
    fp = get_fp(parity_query)

    # An error occurs with the probability of err_rate
    if np.random.randint(0, 101) <= err_rate * 100:
        i = np.random.randint(0, k + 1)
        if i != k:
            # An error occurs among the k outputs of Xs rather than the parity query
            ys_hat[i] = -1

    return ys_hat, fp


# compare the predicition with true labels
def update_paras(ys, ys_hat, ys_hat_hat, err_id, corrs, errs, base_true, parity_true):
    if err_id == -1:
        corrs += k
        for i in range(k):
            if ys[i] == ys_hat[i]:
                base_true += 1
    else:
        errs += k
        for i in range(k):
            if ys[i] == ys_hat_hat[i]:
                parity_true += 1

    return corrs, errs, base_true, parity_true


# calculate Aa, Ad and Ao
def cal_acc(corrs, errs, base_true, parity_true):
    Aa = base_true / corrs
    Ad = parity_true / errs
    Ao = err_rate * Ad + (1 - err_rate) * Aa
    return Aa, Ad, Ao


# record the parameters to calculate the accuary
corrs = 0
errs = 0
base_true = 0
parity_true = 0

for _ in range(epoches):
    cats_num = np.random.randint(0, k + 1)
    dogs_num = k - cats_num
    data_list = np.hstack(
        [np.random.choice(cats_list, cats_num), np.random.choice(dogs_list, dogs_num),]
    )
    # xs: original images
    # ys: true labels
    # ys_hat: predicition with base model
    # ys_hat_hat: predicition with parity model
    xs, ys = get_xs_ys(data_list, cats_num)
    parity_query = encoder.encoder(data_list, 0)
    ys_hat = get_ys_hat(xs)

    ys_hat_hat, fp = query(ys_hat, parity_query)

    # print(ys, end=" , ")
    # print(ys_hat, end=" , ")
    # print(ys_hat_hat)

    err_id = -1
    for i in range(k):
        if ys_hat[i] == -1:
            err_id = i

    if err_id != -1:
        # an error does occur and reconstruct the result with fp
        y_err = fp
        for y in ys_hat:
            if y == 1:
                y_err = y_err - 1
        ys_hat[err_id] = y_err

    corrs, errs, base_true, parity_true = update_paras(
        ys, ys_hat, ys_hat_hat, err_id, corrs, errs, base_true, parity_true
    )

# Aa: the accuary when the unavailability does not occur
# Ad: the accuary when the unavailability does occur
# Ao: the overall accuary
# Ao = err_rate *
print(corrs, errs, base_true, parity_true, sep=" , ")
Aa, Ad, Ao = cal_acc(corrs, errs, base_true, parity_true)
print(Aa, Ad, Ao)
