import numpy as np
import pickle
import os


PROJECT_ROOT_DIR = "../../results"
DATA_ID = "../../results/data"
FIGURE_ID = "../../results/figures"

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


def random_mixed_gaussian(x, n_gaussians=3):
    dim = x.shape[-1]
    mean = np.random.uniform(0, 1, (n_gaussians, 1, dim))

    var = []
    for i in range(n_gaussians):
        var_ = np.random.uniform(-0.001, 0.001, (dim, dim))
        var_[np.diag_indices(dim)] = np.random.uniform(0.005, 0.05, dim)
        var.append(var_)

    alpha = np.random.uniform(-1, 1, n_gaussians)

    y = 0
    for i in range(n_gaussians):
        y += alpha[i] * gaussian(x, mean[i], var[i])

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


def r2(y_pred, y):
    y_avg = np.mean(y)
    return 1 - np.mean((y_pred - y)**2) / np.mean((y - y_avg)**2)
