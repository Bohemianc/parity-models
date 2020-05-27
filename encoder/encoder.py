import cv2
import numpy as np
import tensorflow as tf
from PIL import Image


img_size = 150


def encoder(data_list, draw):
    k = len(data_list)

    # newimg = Image.new("RGB", (img_size, img_size))
    # if k == 4:
    #     for x in range(2):
    #         for y in range(2):
    #             i = x * 2 + y
    #             data = data_list[i]
    #             img = tf.keras.preprocessing.image.load_img(
    #                 data, target_size=(img_size // 2, img_size // 2)
    #             )
    #             newimg.paste(img, (x * img_size // 2, y * img_size // 2))
    # else:
    #     for i in range(k):
    #         data = data_list[i]
    #         img = tf.keras.preprocessing.image.load_img(
    #             data, target_size=(img_size // k, img_size)
    #         )
    #         newimg.paste(img, (0, i * img_size // k))
    # if draw:
    #     newimg.save("tmp.jpg")
    # return tf.keras.preprocessing.image.img_to_array(newimg) / 255.0
    newimg = np.zeros((img_size, img_size, 3), np.uint8)
    if k == 4:
        for x in range(2):
            for y in range(2):
                i = x * 2 + y
                data = data_list[i]
                img = cv2.imread(data)
                img = cv2.resize(img, (img_size // 2, img_size // 2))
                newimg[
                    x * img_size // 2 : (x + 1) * img_size // 2,
                    y * img_size // 2 : (y + 1) * img_size // 2,
                ] = img
    else:
        for i in range(k):
            data = data_list[i]
            img = cv2.imread(data)
            img = cv2.resize(img, (img_size // k, img_size))
            newimg[0:img_size, i * img_size // k : (i + 1) * img_size // k] = img
    if draw:
        cv2.imwrite("tmp.jpg", newimg)
    return newimg / 255.0
