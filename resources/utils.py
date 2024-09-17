import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
import os

spacing = {
    "single_ch":    [14.3, 15.3, 16.3, 17.8, 19.3, 21.3, 23.3, 26.3, 28.3, 31.3, 36.3],
    "18GHz":        [18, 19, 20, 23, 25, 27, 30, 32, 35, 40],
    "17.6GHz":      [18, 19, 20, 21.5, 23, 25, 27, 30, 32, 35, 40],
    "17GHz":        [18, 19, 20, 21.5, 23, 25, 27, 30, 32, 35, 40],
    "16.5GHz":      [18, 19, 20, 21.5, 23, 25, 27, 30, 32, 35, 40],
    "16GHz":        [18, 19, 20, 21.5, 23, 25, 27, 30, 32, 35, 40],
    "15.5GHz":      [20, 21.5, 23, 25, 27, 30, 32, 35, 40],
    "15GHz":        [23, 25, 27, 30, 32, 35, 40],
}

def get_class_array(nb_classes):
    if nb_classes == 2:
        return [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    elif nb_classes == 3:
        return [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2]
    elif nb_classes == 4:
        return [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3]
    elif nb_classes == 5:
        return [0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    elif nb_classes == 11:
        return [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    else:
        return print("Invalid number of classes")

def get_osnr_value(value, nb_classes=2):
    aux = get_class_array(nb_classes)
    dict_osnr_norm = {
        18: aux[0],
        19: aux[1],
        20: aux[2],
        21.5: aux[3],
        23: aux[4],
        25: aux[5],
        27: aux[6],
        30: aux[7],
        32: aux[8],
        35: aux[9],
        40: aux[10],
    }
    return dict_osnr_norm[value]

def get_spacing_value(value):
    dict_spacing = {
        "single_ch": 0,
        "18GHz": 1,
        "17.6GHz": 2,
        "17GHz": 3,
        "16.5GHz": 4,
        "16GHz": 5,
        "15.5GHz": 6,
        "15GHz": 7,
    }
    return dict_spacing[value]

def get_X_extra(x, k0, k1):
    x_extra = (np.eye(8)[x] * k0) * k1
    return x_extra

def load_data(dir, use="regression", nb_classes=11, gaussian_blur=(False, 0), extra_features=True):
    X = []
    y = []
    x_extra = []

    for key, value in spacing.items():
        for osnr in value:
            for i in range(9):
                path = os.path.join(dir, key, f"{osnr}_dB_sample_{i}.png")

                # Get image
                # img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                img = plt.imread(path)

                # Preprocess image
                if gaussian_blur[0]:
                    import cv2
                    img = cv2.GaussianBlur(img, (gaussian_blur[1], gaussian_blur[1]), 0)

                # Append to X
                X.append(img)

                # Append to x_extra
                if extra_features:
                    x_extra.append(get_spacing_value(key))

                # Append to y
                if use == "regression":
                    y.append(osnr if key != "single_ch" else osnr + 3.7)
                elif use == "classification":
                    y.append(get_osnr_value(osnr, nb_classes) if key != "single_ch" else get_osnr_value(osnr + 3.7, nb_classes))
                else:
                    return print("Invalid use parameter")
                
    if use == "classification":
        y = to_categorical(y, num_classes=nb_classes)
    elif use == "regression":
        y = np.array(y)
        y = y.reshape(y.shape[0], 1)
    else:
        return print("Invalid use parameter")
                
    if extra_features:
        return np.array(X), np.array(y), get_X_extra(np.array(x_extra), -.5, 2.)
    else:
        return np.array(X), np.array(y)
