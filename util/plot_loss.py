"""
Compare training and validation loss between models.
"""

import matplotlib.pyplot as plt
import numpy as np


MODELS = ['9800_epoch100/5x64-1x32-3', '20k_epoch88/5x64-1x32-3', '20k_complex_epoch67/7x64-3']
TRAIN_LOSS_PATH = 'path/to/{model}_best/train_loss.txt'
VAL_LOSS_PATH = 'path/to/{model}_best/val_loss.txt'


def plot_loss():
    """
    Plot training and validation loss for each model.
    """

    for i, model in enumerate(MODELS):
        train_loss = np.loadtxt(TRAIN_LOSS_PATH.format(model=model))
        val_loss = np.loadtxt(VAL_LOSS_PATH.format(model=model))
        assert train_loss.shape == val_loss.shape
        epochs = np.arange(1, len(train_loss) + 1)

        plt.plot(epochs, train_loss, c=f'C{i}', ls='--', label=f'{model} training')
        plt.plot(epochs, val_loss, c=f'C{i}', label=f'{model} validation')

        # # Plot best only
        # best_idx = np.argmin(val_loss)
        # plt.scatter(epochs[best_idx], val_loss[best_idx], c=f'C{i}', label=f'{model} validation')

    plt.yscale('log')
    plt.legend()
    plt.show()
