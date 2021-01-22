from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def show_release_point(x, y, z, color):
    fig_release_point = plt.figure()
    ax = Axes3D(fig_release_point)
    ax.scatter(x, y, z, color)
    plt.show()


def show_pitch_location(x, z, color):
    plt.scatter(x, z, c=color)
    plt.show()

