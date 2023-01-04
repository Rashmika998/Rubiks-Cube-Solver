import sys
import numpy as np
from Cube import Cube
from Solver import Solver
from SolutionGallery import SolutionGallery
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.patches import Rectangle
from matplotlib.image import AxesImage
from matplotlib import get_backend
from rubiks_cube_face_recognition import detect_colors

class ClickableRectangle:
    def __init__(self, fig, ax, colors_cube, f, i_f, j_f):
        self.f = f
        self.i_f = i_f
        self.j_f = j_f

        self.colors_cube = colors_cube
        self.x = i_f
        self.y = j_f
        self.color = self.colors_cube[f, i_f, j_f]

        self.fig = fig
        self.ax = ax

        self.rect = Rectangle( 
            (self.x, self.y),
            width=1,
            height=1,
            facecolor=self.color,
            linewidth=2,
            edgecolor='k',
            picker=True
        )

        self.patch = self.ax.add_patch(self.rect)

        self.active = False
        clicker = self.fig.canvas.mpl_connect(
            'button_press_event', lambda e: self.onclick(e))
        presser = self.fig.canvas.mpl_connect(
            'key_press_event', lambda e: self.keypress(e))

    def onclick(self, event):
        # Was the click on the same axis as the rectangle?
        if event.inaxes != self.rect.axes:
            self.active = False
            return

        # Was the click inside the rectangle?
        contains, attrd = self.rect.contains(event)
        if not contains:
            self.active = False
            return

        # Only concerned with double click events
        if event.dblclick:
            # Set active
            self.active = True

    def keypress(self, event):
        if not self.active:
            return

        elif event.key in ['w', 'W']:
            self.color = 'white'
        elif event.key in ['o', 'O']:
            self.color = 'orange'
        elif event.key in ['r', 'R']:
            self.color = 'red'
        elif event.key in ['g', 'G']:
            self.color = 'green'
        elif event.key in ['b', 'B']:
            self.color = 'blue'
        elif event.key in ['y', 'Y']:
            self.color = 'yellow'
        elif event.key == 'enter':
            self.active = False

        self.colors_cube[self.f, self.i_f, self.j_f] = self.color

        self.patch.set_facecolor(self.color)
        self.fig.canvas.draw()


class RectContainer:
    def __init__(self, fig, ax_squares, f, colors_cube):
        self.fig = fig
        self.ax_squares = ax_squares
        self.f = f
        self.colors_cube = colors_cube

        # Plot the colors on the right of the image
        self.clickable_rects = []
        for s in range(9):
            i_f = s % 3
            j_f = int(s / 3)
            cr = ClickableRectangle(
                self.fig, self.ax_squares, self.colors_cube, self.f, i_f, j_f)
            self.clickable_rects.append(cr)

        self.ax_squares.set_xlim(0, 3)
        self.ax_squares.set_ylim(0, 3)
        self.ax_squares.axis('equal')
        self.ax_squares.axis('off')
        self.fig.canvas.draw()

    def update_squares(self):
        # Plot the colors on the right of the image
        for s in range(9):
            i_f = s % 3
            j_f = int(s / 3)
            cr = self.clickable_rects[s]
            cr.patch.set_facecolor(self.colors_cube[self.f, i_f, j_f])

    # def onclick(self, event):
    #     if event.inaxes != self.ax_img.axes:
    #         return

    #     if event.dblclick:

    #         pred_colors_yuv=['red', 'green', 'red', 'red', 'orange', 'red', 'blue', 'white', 'red', 
    #                             'white', 'orange', 'green', 'yellow', 'green', 'red', 'orange', 'orange', 'red', 
    #                             'green', 'blue', 'white', 'yellow', 'red', 'green', 'orange', 'white', 'orange', 
    #                             'yellow', 'blue', 'green', 'orange', 'blue', 'red', 'white', 'orange', 'blue', 
    #                             'yellow', 'green', 'white', 'white', 'white', 'white', 'yellow', 'blue', 'green', 
    #                             'blue', 'yellow', 'red', 'green', 'yellow', 'yellow', 'yellow', 'blue','orange']
    #         pred_colors_yuv2 = np.array(pred_colors_yuv).reshape((1, 3, 3))

    #         self.colors_cube[self.f, :, :] = pred_colors_yuv2


    #         #self.ax_img.imshow(self.faces)
    #         self.update_squares()
    #         self.fig.canvas.draw()


def onpick(event):
    if isinstance(event.artist, Rectangle):
        patch = event.artist
        # print('onpick patch:', patch.get_path())
        patch.set_edgecolor('lime')
        event.canvas.draw()

    elif isinstance(event.artist, AxesImage):
        im = event.artist
        A = im.get_array()
        # print('onpick image', A.shape)


def check_images(colors_cube):
    fig, axs = plt.subplots(1, 6, figsize=(24, 6))

    for f in range(6):

        RectContainer(
            fig,
            axs[f],
            f,
            colors_cube
        )

    fig.tight_layout()

    return


def main():
    pred_colors_yuv=detect_colors()
    colors_cube = np.array(pred_colors_yuv).reshape((6, 3, 3))

    # Inspect / adjust results if necessary. This step can modify pred_colors_yuv2.
    check_images(colors_cube)
     # Define the cube using the updated colors
    plt.suptitle("If any squares have mistakes, double click on the square and type the first letter of the correct color\n(e.g. If the square should have been blue, type 'b')\nEnter 'q' once the correct colors have been entered")
    plt.show()


    c = Cube(colors_cube)

    # Solve and retain moves
    initial_state = c.export_state() #set the initial state of the cube
    s = Solver(c)
    s.solve()
    solve_moves = c.recorded_moves

    # Display the solution
    sg = SolutionGallery(initial_state, solve_moves)
    plt.show()


if __name__ == '__main__':
    main()
