from Solver import Solver
from Cube import Cube
from plot_annotations import add_move_annotation
import matplotlib.pyplot as plt


def opposite_move(move):
    if '-' in move:
        move = move[1:]
    else:
        move = '-' + move
    return move


class SolutionGallery:
    def __init__(self, initial_state, solve_moves, i_start=0):
        self.i = 0
        self.n_moves = len(solve_moves)
        self.c = Cube()
        self.c.set_state(initial_state)
        self.solve_moves = solve_moves

        # Put cube in state after i_start moves.
        while self.i < i_start:
            self.c.perform_move(self.solve_moves[self.i])
            self.i += 1

        # Figure and axes. Place in initial state
        self.fig, _ = plt.subplots(figsize=(6, 6))
        plt.axis('off')
        self.ax1 = self.fig.add_subplot(111, projection='3d')
        self.ax1 = self.c.cube_plot(
            ax=self.ax1, title_str='Initial State\n Use left/right arrow keys to navigate')

        # Set up connection to detect keys
        self.mpl_id = self.fig.canvas.mpl_connect(
            'key_press_event', self.handle_keypress)

    def show(self):
        plt.show()

    def handle_keypress(self, event):

        if event.key == 'right':
            if self.i < self.n_moves:
                move = self.solve_moves[self.i]
                self.c.cube_plot(ax=self.ax1, title_str=move.replace('1', ''))
                add_move_annotation(self.ax1, move)
                self.c.perform_move(move)
                self.i = self.i + 1
            else:
                self.c.cube_plot(
                    ax=self.ax1, title_str='Final State\n(press q to quit)'
                )
            plt.draw()

        if event.key == 'left':
            if self.i > 0:
                self.i = self.i - 1

                # Undo the last move to put cube in previous state
                self.c.perform_move(opposite_move(self.solve_moves[self.i]))

                # Display the next move
                move = self.solve_moves[self.i]
                self.c.cube_plot(ax=self.ax1, title_str=move.replace('1', ''))
                add_move_annotation(self.ax1, move)
            else:
                self.c.cube_plot(
                    ax=self.ax1, title_str='Initial State'
                )
            plt.draw()

        return event


if __name__ == '__main__':
    # Create a random cube
    c = Cube()
    c.randomize_state()
    initial_state = c.export_state()

    # Solve and retain moves
    s = Solver(c)
    s.solve()

    # Display the solution
    sg = SolutionGallery(s.initial_state, s.solve_moves)
    sg.show()
