from model import GAN
from matplotlib import pyplot as plt
import tensorflow as tf
from data import *


num_points = 64000


def main():
    d = gaussian_data
    with tf.Session() as sess:
        model = GAN(sess, d, [8, 8, 8], [6, 6, 6], x_dim=1, z_dim=1, lr=0.0001, k=1, std=3.0, mean=3.0)
        model.train(100000, 100)
        decision, gd, dd, x = model.sample(num_points=num_points)
        plot(decision, gd, dd, x)


def plot(decision, gd, dd, x):
    """
    Plot the decision boundary, generator's distribution, and read data distribution
    """
    bins = len(dd)
    x_axis = np.linspace(np.min(x), np.max(x), bins)
    plt.plot(x_axis, gd)
    plt.plot(x_axis, dd)
    # plt.plot(np.linspace(np.min(x), np.max(x), len(decision)), decision)
    plt.show()

if __name__ == '__main__':
    main()