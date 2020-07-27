import qiskit.quantum_info
import numpy as np 
import torch
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot
import matplotlib.pyplot as plt 
from xgboost import XGBClassifier
from confusion_learning.energy import energy

def step_gen(X, y, transition=0.,k = 1, slope=100):
    X = X.reshape(-1, 1)
    noise = 0.001 * np.random.randn(*X.shape)

    return 1 / (1 + np.exp(-slope * (X - k * y - transition + noise))) + noise

def energy_gen(x, h, n_thetas=100):
    # X is a list of impurity concentration
    # h is a parameter of magnetic field

    # here we generate range of energies with given X and h
    # (you can change other parameters of task in confusion_learning/energy.py
    Thetas = np.sort(np.random.rand(n_thetas) * 2 * np.pi)
    # Thetas = np.linspace(0, 2 * np.pi, n_thetas)
    Energies = energy(Thetas, x, h)
    e_max = np.max(Energies)
    e_min = np.min(Energies)

    # return np.argmin((Energies - e_min) / (e_max - e_min))
    return (Energies - e_min) / (e_max - e_min)

def data_labeling(data, params, p_expect):
    labels = (params > p_expect).astype('float')
    # data_mean = np.mean(data, axis=0)
    # data_std = np.std(data, axis=0)
    # data = (data - data_mean) / (data_std + 0.01)

    return data, labels


def XGB_learning(data, labels):
    data_train, data_test, labels_train, labels_test = \
        train_test_split(data, labels, test_size=0.2, random_state=7)

    # fit model no training data

    model = XGBClassifier()
    eval_set = [(data_train, labels_train), (data_test, labels_test)]
    model.fit(data_train, labels_train, eval_metric=["error", "logloss"], eval_set=eval_set, verbose=False)

    labels_pred = model.predict(data_test)
    predictions = [round(value) for value in labels_pred]

    results = model.evals_result()
    accuracy = accuracy_score(labels_test, predictions)
    learn_curve = 1 - np.array(results['validation_1']['error'])
    # print('Accuracy = ', accuracy)

    return accuracy, learn_curve

def plot_learn_curves_cut(learn_curves, p_guess, x_true):
    fig = plt.figure()
    plt.rc('font', family='serif')
    cmap = plt.get_cmap('twilight')
    ax = fig.add_subplot(1, 1, 1)
    fig.set_size_inches(6, 3)
    leng = len(learn_curves[0, :])
    for i in range(leng):
        ax.plot(p_guess, learn_curves[:, i], color=cmap(float(i)/leng))
    ax.axvline(x=x_true)
    ax.set_xlabel('p', fontsize=15)
    ax.set_ylabel('Accuracy', fontsize=15)
    ax.grid()
    plt.show()


def w_shape_gen(data, params):
    w_data, learn_curves = [], []

    for p in params:
        data, labels = data_labeling(data, params, p)
        acc, curve = XGB_learning(data, labels)
        w_data.append(acc)
        learn_curves.append(curve)
    return w_data, learn_curves


def mainloop(X, H, n_thetas=100, n_samples=10):

    Z = np.zeros((X.shape[0], H.shape[0]))

    for i, h in tqdm(enumerate(H)):
        w_data_stack = []
        for _ in tqdm(range(n_samples)):
            # print('------- w-shape sample number =', i, '-------')

            raw_data = np.array([np.argmin(energy_gen(x, h, n_thetas)) for x in X]).reshape(-1, 1)
            w_data, learn_curves = w_shape_gen(raw_data, X)

            w_data_stack.append(w_data)
            # learn_curves_data.append(learn_curves)
        w_shape = np.mean(w_data_stack, axis=0)
        Z[:, i] = w_shape #sry, I think this could be better

        # w_bar = np.std(w_data_stack, axis=0)
        # learn_curves = np.mean(learn_curves_data, axis=0)

    return Z



if __name__ == '__main__':
    X = np.linspace(0, 1, 50)

    h = 10 ** 3
    Energies = energy_gen(X, h)
    print(Energies)



