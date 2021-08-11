import numpy as np
import pickle
import os


PROJECT_ROOT_DIR = "../../results"
DATA_ID = "../../results/data"
FIGURE_ID = "../../latex/figures"

if not os.path.exists(PROJECT_ROOT_DIR):
    os.mkdir(PROJECT_ROOT_DIR)
if not os.path.exists(FIGURE_ID):
    os.makedirs(FIGURE_ID)
if not os.path.exists(DATA_ID):
    os.makedirs(DATA_ID)


def image_path(fig_id):
    return os.path.join(FIGURE_ID, fig_id)


def data_path(data_id):
    return os.path.join(DATA_ID, data_id)


def identity(func):
    return func


def saver(object, filename):
    pickle.dump(object, open(filename, "wb"))


def loader(filename):
    object = pickle.load(open(filename, "rb"))

    return object


def unpack_list(list_):
    list_flat = []
    for l in list_:
        list_flat.append(l.flatten())

    list_flat = np.concatenate(list_flat).reshape(-1, 1)

    return list_flat


def gaussian(x, mean, var):
    if type(mean) == float:
        mean = np.array([[mean]])

    if type(var) == float:
        var = np.array([[var]])

    var_inv = np.linalg.inv(var)
    diag = np.diag((x - mean) @ var_inv @ (x - mean).T).reshape(-1, 1)
    y = np.exp(-0.5 * diag)

    return y


def scaler(x, mode="uniform", a=0, b=np.pi):
    if mode == "uniform":
        x = x - np.min(x, axis=0)
        x = (b - a) * x / np.max(x, axis=0)
        x = x + a
    if mode == "standard":
        x = (x - np.mean(x, axis=0)) / np.std(x, axis=0)

    return x


def generate_meshgrid(x):

    x = np.meshgrid(*x)
    x = [np.ravel(x_).reshape(-1, 1) for x_ in x]
    x = np.hstack(x)

    return x


def r2(models, x, y):
    y_avg = np.mean(y)

    if type(models) != list:
        models = [models]

    r2_scores = []
    for model in models:
        y_pred = model.predict(x)
        r2 = 1 - np.mean((y_pred - y)**2) / np.mean((y - y_avg)**2)
        r2_scores.append(r2)

    return sum(r2_scores) / len(r2_scores)


def accuracy(y_pred, y):
    acc = np.mean(np.round(y_pred) == y)
    return acc


def generate_1D_mixed_gaussian():
    n = 100
    x = np.linspace(0, 1, n).reshape(-1, 1)
    y = gaussian(x, 0.2, 0.01) - gaussian(x, 0.5, 0.01) + \
        gaussian(x, 0.8, 0.01)
    y = scaler(y, a=0, b=1)
    return (x, y)


def generate_2D_mixed_gaussian():
    n = 12
    x = np.linspace(0, 1, n)
    x = generate_meshgrid([x, x])

    mean1 = np.array([[0.2, 0.8]])
    mean2 = np.array([[0.5, 0.8]])
    mean3 = np.array([[0.8, 0.8]])
    mean4 = np.array([[0.2, 0.5]])
    mean5 = np.array([[0.5, 0.5]])
    mean6 = np.array([[0.8, 0.5]])
    mean7 = np.array([[0.2, 0.2]])
    mean8 = np.array([[0.5, 0.2]])
    mean9 = np.array([[0.8, 0.2]])
    var = np.array([[0.01, 0], [0, 0.01]])

    y = gaussian(x, mean1, var) - gaussian(x, mean2, var) +\
        gaussian(x, mean3, var) - gaussian(x, mean4, var) +\
        gaussian(x, mean5, var) - gaussian(x, mean6, var) +\
        gaussian(x, mean7, var) - gaussian(x, mean8, var) +\
        gaussian(x, mean9, var)

    y = scaler(y, a=0, b=1)
    return (x, y)


def generate_3D_mixed_gaussian():
    n = 6
    x = np.linspace(0, 1, n)
    x = generate_meshgrid([x, x, x])

    mean1 = np.array([[0.25, 0.25, 0.25]])
    mean2 = np.array([[0.25, 0.25, 0.75]])
    mean3 = np.array([[0.25, 0.75, 0.75]])
    mean4 = np.array([[0.25, 0.75, 0.25]])
    mean5 = np.array([[0.75, 0.25, 0.25]])
    mean6 = np.array([[0.75, 0.25, 0.75]])
    mean7 = np.array([[0.75, 0.75, 0.75]])
    mean8 = np.array([[0.75, 0.75, 0.25]])

    var = np.array([[0.02, 0, 0], [0, 0.02, 0], [0, 0, 0.02]])

    y = gaussian(x, mean1, var) - gaussian(x, mean2, var) + \
        gaussian(x, mean3, var) - gaussian(x, mean4, var) - \
        gaussian(x, mean5, var) + gaussian(x, mean6, var) - \
        gaussian(x, mean7, var) + gaussian(x, mean8, var)

    y = scaler(y, a=0, b=1)

    return (x, y)
