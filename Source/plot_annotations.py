import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

from Cube import Cube


def basic_triangle(D):
    width = 0.75
    nh = 3
    n = 2 * nh + 1
    b_tri = 0.5 * width
    h_tri = b_tri * np.sqrt(3)

    X_tri = np.linspace(-b_tri, b_tri, n)
    X_tri = np.tile(X_tri, (n, 1))
    Y_tri = np.zeros_like(X_tri)

    for i in range(nh+1):
        Y_tri[:, i] = np.linspace(
            0, (h_tri / b_tri) * (b_tri + X_tri[0, i]), n) - h_tri / 3
        Y_tri[:, n-i-1] = Y_tri[:, i]

    X_tri = X_tri + D[0]
    Y_tri = Y_tri + D[1]
    Z_tri = D[2] * np.ones_like(X_tri)

    return X_tri, Y_tri, Z_tri


def basic_arrow():
    r = 3.0
    width = 0.125
    theta = np.linspace(0, np.pi, 30)
    X = np.linspace(-width, width, 10)
    theta, X = np.meshgrid(theta, X)
    X, Y, Z = 4.5 + X, r*np.sin(theta), r*np.cos(theta)

    nh = 10
    n = 2 * nh + 1
    w_tri = 3.0 * width
    phi = (np.pi / 2) * (75 / 45)
    h_tri = w_tri * np.arctan(phi)
    X_tri = np.linspace(-w_tri, w_tri, n)
    X_tri = np.tile(X_tri, (n, 1))
    Y_tri = np.zeros_like(X_tri)
    for i in range(nh+1):
        Y_tri[:, i] = np.linspace(
            0, -(h_tri / w_tri) * (w_tri + X_tri[0, i]), n)
        Y_tri[:, n-i-1] = np.linspace(0, -(h_tri / w_tri)
                                      * (w_tri + X_tri[0, i]), n)
    X_tri = 4.5 + X_tri
    Z_tri = - r * np.ones_like(X_tri)

    return X, Y, Z, X_tri, Y_tri, Z_tri


def add_move_annotation(ax, move):
    if '-' in move:
        orientation = 'CCW'
    else:
        orientation = 'CW'
    move_type = move[-1]

    if move_type in ['U', 'D', 'L', 'R', 'F', 'B']:
        for i in range(3):
            X_tri, Y_tri, Z_tri = basic_triangle(
                D=np.array([2, -2 + 2*i, 4.5]))

            if orientation == 'CCW':
                Y_tri = -Y_tri

            if move_type == 'R':
                ax.plot_surface(X_tri,  Y_tri,  Z_tri, color='k',
                                edgecolor='none', shade=False)
                ax.plot_surface(X_tri, -Z_tri,  Y_tri, color='k',
                                edgecolor='none', shade=False)
                ax.plot_surface(X_tri, -Y_tri, -Z_tri, color='k',
                                edgecolor='none', shade=False)
                ax.plot_surface(X_tri,  Z_tri, -Y_tri, color='k',
                                edgecolor='none', shade=False)

            elif move_type == 'L':
                ax.plot_surface(-X_tri, -Y_tri,  Z_tri, color='k',
                                edgecolor='none', shade=False)
                ax.plot_surface(-X_tri, -Z_tri, -Y_tri, color='k',
                                edgecolor='none', shade=False)
                ax.plot_surface(-X_tri,  Y_tri, -Z_tri, color='k',
                                edgecolor='none', shade=False)
                ax.plot_surface(-X_tri,  Z_tri,  Y_tri, color='k',
                                edgecolor='none', shade=False)

            elif move_type == 'F':
                ax.plot_surface(Y_tri, -X_tri,  Z_tri, color='k',
                                edgecolor='none', shade=False)
                ax.plot_surface(Z_tri, -X_tri, -Y_tri, color='k',
                                edgecolor='none', shade=False)
                ax.plot_surface(-Y_tri, -X_tri, -Z_tri, color='k',
                                edgecolor='none', shade=False)
                ax.plot_surface(-Z_tri, -X_tri,  Y_tri, color='k',
                                edgecolor='none', shade=False)

            elif move_type == 'B':
                ax.plot_surface(-Y_tri, X_tri,  Z_tri, color='k',
                                edgecolor='none', shade=False)
                ax.plot_surface(Z_tri, X_tri,  Y_tri, color='k',
                                edgecolor='none', shade=False)
                ax.plot_surface(Y_tri, X_tri, -Z_tri, color='k',
                                edgecolor='none', shade=False)
                ax.plot_surface(-Z_tri, X_tri, -Y_tri, color='k',
                                edgecolor='none', shade=False)

            elif move_type == 'U':
                ax.plot_surface(-Y_tri, -Z_tri, X_tri, color='k',
                                edgecolor='none', shade=False)
                ax.plot_surface(Z_tri, -Y_tri, X_tri, color='k',
                                edgecolor='none', shade=False)
                ax.plot_surface(Y_tri,  Z_tri, X_tri, color='k',
                                edgecolor='none', shade=False)
                ax.plot_surface(-Z_tri,  Y_tri, X_tri, color='k',
                                edgecolor='none', shade=False)

            elif move_type == 'D':
                ax.plot_surface(Y_tri, -Z_tri, -X_tri,  color='k',
                                edgecolor='none', shade=False)
                ax.plot_surface(Z_tri,  Y_tri, -X_tri,  color='k',
                                edgecolor='none', shade=False)
                ax.plot_surface(-Y_tri,  Z_tri, -X_tri,
                                color='k', edgecolor='none', shade=False)
                ax.plot_surface(-Z_tri, -Y_tri, -X_tri,
                                color='k', edgecolor='none', shade=False)

    elif move_type in ['X', 'Y', 'Z']:
        # Make the basic arrow
        X, Y, Z, X_tri, Y_tri, Z_tri = basic_arrow()

        if move_type == 'X':
            if orientation == 'CCW':
                Z = -Z
                Z_tri = -Z_tri
            ax.plot_surface(X, Y, Z, color='b', edgecolor='none')
            ax.plot_surface(X_tri, Y_tri, Z_tri, color='b', edgecolor='none')
            ax.plot_surface(-X, Y, Z, color='b', edgecolor='none')
            ax.plot_surface(-X_tri, Y_tri, Z_tri, color='b', edgecolor='none')

        elif move_type == 'Y':
            X_tri, Y_tri, Z_tri = Y_tri, X_tri, Z_tri
            X, Y, Z = Y, X, Z
            if orientation == 'CCW':
                X = -X
                X_tri = -X_tri
            ax.plot_surface(X, Y, Z, color='b', edgecolor='none')
            ax.plot_surface(X_tri, Y_tri, Z_tri, color='b', edgecolor='none')
            ax.plot_surface(X, -Y, Z, color='b', edgecolor='none')
            ax.plot_surface(X_tri, -Y_tri, Z_tri, color='b', edgecolor='none')

        elif move_type == 'Z':
            X_tri, Y_tri, Z_tri = Y_tri, Z_tri, X_tri
            X, Y, Z = Y, Z, X
            if orientation == 'CCW':
                X = -X
                X_tri = -X_tri
            ax.plot_surface(X, Y, Z, color='b', edgecolor='none')
            ax.plot_surface(X_tri, Y_tri, Z_tri, color='b', edgecolor='none')
            ax.plot_surface(X, Y, -Z, color='b', edgecolor='none')
            ax.plot_surface(X_tri, Y_tri, -Z_tri, color='b', edgecolor='none')

    #add_arrow(ax, move)


if __name__ == '__main__':
    c = Cube()
    initial_state = c.export_state()

    for move_type in ['U', 'D', 'R', 'L', 'F', 'B', 'X', 'Y', 'Z']:
        for orientation in ['1', '-1']:
            move = orientation + move_type
            fig, ax = plt.subplots(figsize=(8, 4))
            plt.axis('off')
            ax = fig.add_subplot(121, projection='3d')
            c.set_state(initial_state)
            c.cube_plot(ax=ax, title_str=move)
            add_move_annotation(ax, move)
            ax = fig.add_subplot(122, projection='3d')
            c.perform_move(move)
            c.cube_plot(ax=ax)
    plt.show()
