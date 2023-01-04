import numpy as np
import matplotlib.pyplot as plt
from Cube import (
    Cube, face_dict, face_normals, opposite_face,
)


class Solver:
    def __init__(self, c):
        self.c = c
        self.initial_state = self.c.export_state()
        self.recorded_moves = []
        self.solve_moves = []

    def white_face_to_top(self):
        f = face_dict[self.c.center_piece_idx('white')[0]]
        self.c.rotate_cube_face_to_face(f, 'up')

    def remove_ep_from_right_edge(self, edge_to_switch):
        self.c.rotate_face_edge_to_edge('right', edge_to_switch, 'down')
        self.c.rotate_down(-1)
        self.c.rotate_face_edge_to_edge('right', 'down', edge_to_switch)

    def solve_white_cross(self):
        # Ensure white cross is on top
        self.white_face_to_top()

        for ep_to_solve in self.c.edge_pieces_matching_color('white'):
            adj_ep_to_solve = ep_to_solve.adj_pieces[0]

            if not self.c.edge_piece_solved(ep_to_solve):
                # use the adjacent piece to identify the target face color
                target_adj_face_color = adj_ep_to_solve.color_name
                target_face_piece = self.c.center_piece_matching_color(
                    target_adj_face_color)

                if ep_to_solve.face_name in ['front', 'back', 'left', 'right']:
                    self.c.rotate_cube_face_to_face(
                        ep_to_solve.face_name, 'right')

                    # Target color is on the right, so we're already
                    # on the correct side
                    if self.c.face_color('right') == adj_ep_to_solve.color_name:
                        self.c.rotate_face_edge_to_edge(
                            'right', adj_ep_to_solve.face_name, 'front'
                        )

                    # Target color is on another side, so move piece to correct
                    # side and flip to right
                    else:
                        self.remove_ep_from_right_edge(
                            adj_ep_to_solve.face_name)
                        self.c.rotate_face_edge_to_edge(
                            'down', 'front', target_face_piece.face_name)
                        self.c.rotate_cube_face_to_face(
                            target_face_piece.face_name, 'right')
                        self.c.rotate_face_edge_to_edge(
                            'right', 'down', 'front')

                    # the piece is now on the right face, front edge. solve
                    self.c.perform_move_list([
                        'U', '-1F', '-1U'
                    ])

                else:
                    self.c.rotate_cube_face_to_face(
                        adj_ep_to_solve.face_name,
                        'right'
                    )

                    if ep_to_solve.face_name == 'up':
                        self.c.rotate_right(2)

                    self.c.rotate_face_edge_to_edge(
                        'down',
                        adj_ep_to_solve.face_name,
                        target_face_piece.face_name
                    )

                    self.c.rotate_cube_face_to_face(
                        target_face_piece.face_name,
                        'right'
                    )

                    self.c.rotate_right(2)

                if not self.c.edge_piece_solved(ep_to_solve):
                    self.c.cube_plot(
                        title_str='Something went wrong with the current edge piece.'
                    )

    def solve_white_corners(self):
        # White corner pieces:
        white_corner_pieces = self.c.corner_pieces_matching_color('white')

        for i in range(4):
            # We are going to solve the right/front corner which has
            # the following colors:
            colors_to_solve = [
                self.c.face_color('right'),
                self.c.face_color('front')
            ]

            # identify the corner we need to send to the target_corner
            for piece_to_solve in white_corner_pieces:
                if piece_to_solve.adjacent_color_names() == set(colors_to_solve):
                    break

            # all points will move here
            pt_corner_to_solve = np.array([2, -2, 3], dtype='int')
            target_corner = set(['front', 'right', 'up'])

            # if already solved don't do anything
            if self.c.corner_piece_solved(piece_to_solve):
                pass

            # piece is unsolved, but in the target corner. this will move it
            # to front/down/right position
            elif piece_to_solve.corner() == target_corner:
                self.c.perform_move_list(['-1R', '-1D', 'R', 'D'])

            # piece is unsolved but in some other up corner.
            # move it to front,down,right position
            elif 'up' in piece_to_solve.corner():
                tmp = sorted(list(piece_to_solve.corner()))
                tmp.remove('up')

                if tmp[0] != 'right':
                    face_to_rotate = tmp[0]
                else:
                    face_to_rotate = tmp[1]

                corner_start = piece_to_solve.corner()
                corner_end = tmp
                corner_end.append('down')
                corner_end = set(corner_end)

                self.c.rotate_face_corner_to_corner(
                    face_to_rotate, corner_start, corner_end
                )

                self.c.rotate_face_corner_to_corner(
                    'down', corner_end, set(['front', 'right', 'down'])
                )

                self.c.rotate_face_corner_to_corner(
                    face_to_rotate, corner_end, corner_start
                )

            # piece is on down edge. move to front,right,down position
            if 'down' in piece_to_solve.corner():
                self.c.rotate_face_corner_to_corner(
                    'down', piece_to_solve.corner(), set(['front', 'right', 'down'])
                )

            # case that white is on front:
            if piece_to_solve.face_name == 'front':
                self.c.perform_move_list([
                    '-1D', '-1R', 'D', 'R'
                ])

            # case that white is on right
            elif piece_to_solve.face_name == 'right':
                self.c.perform_move_list([
                    '-1R', '-1D', 'R'
                ])

            # case that white is on down face
            elif not self.c.corner_piece_solved(piece_to_solve):
                self.c.perform_move_list([
                    '-1R', '-2D', 'R', 'D', '-1R', '-1D', 'R'
                ])

            # Did something go wrong?
            if not self.c.corner_piece_solved(piece_to_solve):
                self.c.cube_plot(
                    title_str='Something went wrong with the current corner piece.'
                )
                plt.show()

            # Rotate next face to the front
            self.c.rotate_z(1)

    def remove_middle_edge_piece(self, e):
        self.c.rotate_cube_face_to_face(e, 'front')
        self.c.perform_move_list([
            'U', 'R', '-1U', '-1R', '-1U', '-1F', 'U', 'F'
        ])
        self.c.rotate_cube_face_to_face('front', e)

    def solve_middle_edges(self):
        # Put white on bottom
        self.c.rotate_cube_face_to_face(
            face_dict[self.c.center_piece_idx('white')[0]], 'down'
        )

        # We will solve corners one-by-one clock-wise
        for i in range(4):
            colors_to_solve = [
                self.c.face_color('front'), self.c.face_color('right')
            ]

            ep_to_solve = self.c.edge_piece_matching_colors(
                colors_to_solve[0], colors_to_solve[1]
            )

            if not self.c.edge_piece_solved(ep_to_solve):

                if 'up' not in ep_to_solve.edge():
                    if ep_to_solve.edge() == set(['right', 'back']):
                        self.remove_middle_edge_piece('right')
                    elif ep_to_solve.edge() == set(['left', 'back']):
                        self.remove_middle_edge_piece('back')
                    elif ep_to_solve.edge() == set(['left', 'front']):
                        self.remove_middle_edge_piece('left')
                    elif ep_to_solve.edge() == set(['front', 'right']):
                        self.remove_middle_edge_piece('front')

                if ep_to_solve.face_name == 'up':
                    ep_to_solve = ep_to_solve.adj_pieces[0]
                starting_edge = ep_to_solve.face_name

                if ep_to_solve.color_name == self.c.face_color('front'):
                    self.c.rotate_face_edge_to_edge(
                        'up', starting_edge, 'front'
                    )
                    self.c.perform_move_list([
                        'U', 'R', '-1U', '-1R', '-1U', '-1F', 'U', 'F'
                    ])

                elif ep_to_solve.color_name == self.c.face_color('right'):
                    self.c.rotate_face_edge_to_edge(
                        'up', starting_edge, 'right')
                    self.c.rotate_cube_face_to_face('right', 'front')
                    self.c.perform_move_list([
                        '-1U', '-1L', 'U', 'L', 'U', 'F', '-1U', '-1F'
                    ])
                    self.c.rotate_cube_face_to_face('front', 'right')

            self.c.rotate_cube_face_to_face('right', 'front')

    def solve_yellow_cross(self):
        yellow_edge_pieces = self.c.edge_pieces_matching_color('yellow')

        while len([ep for ep in yellow_edge_pieces if ep.face_name == 'up']) < 4:
            yellow_cross_pieces = [
                ep for ep in yellow_edge_pieces if ep.face_name == 'up'
            ]
            n_yellow_cross_in_place = len(yellow_cross_pieces)

            # None placed, state 2 from manual
            if n_yellow_cross_in_place == 0:
                # solve cross w/ State 2 method
                self.c.perform_move_list([
                    'F', 'U', 'R', '-1U', '-1R', '-1F'
                ])

            else:
                ep_adj1 = yellow_cross_pieces[0].adj_pieces[0]
                ep_adj2 = yellow_cross_pieces[1].adj_pieces[0]

                n1 = face_normals[ep_adj1.face_name]
                n2 = face_normals[ep_adj2.face_name]
                orientation = np.dot(
                    np.cross(n1, n2), np.array([0, 0, 1], dtype='int'))

                # Two up yellow edge pieces are in a line, state 4 from manual,
                # after rotation
                if orientation == 0:
                    # Line is perpendicular to front in this case. Needs to be
                    # parallel for State 4.
                    if ep_adj1.face_name in ['front', 'back']:
                        self.c.rotate_face_edge_to_edge('up', 'front', 'right')

                    # solve cross w/ State 4 method
                    self.c.perform_move_list([
                        'F', 'R', 'U', '-1R', '-1U', '-1F'
                    ])

                # Two up yellow edge pieces are not in a line, state 3 from
                # the manual, after rotation.
                else:
                    # If n1,n2 is ccw oriented, then we can rotate ep_adj1's
                    # face to back to get state 3
                    if orientation > 0:
                        self.c.rotate_face_edge_to_edge(
                            'up', ep_adj1.face_name, 'back'
                        )

                    # Otherwise, we should rotate ep_adj2's face to back to get state 3.
                    if orientation < 0:
                        self.c.rotate_face_edge_to_edge(
                            'up', ep_adj2.face_name, 'back'
                        )

                    # solve cross w/  State 3 method
                    self.c.perform_move_list([
                        'F', 'U', 'R', '-1U', '-1R', '-1F'
                    ])

    def non_up_yellow_corner_orientation(self, cp):
        n1 = face_normals[cp.face_name]
        n2 = [face_normals[cp_adj.face_name]
              for cp_adj in cp.adj_pieces if cp_adj.face_name != 'up'][0]
        orientation = np.dot(np.cross(n1, n2), face_normals['up'])
        return orientation

    def solve_yellow_corners(self):
        while len([ep for ep in self.c.corner_pieces_matching_color('yellow')
                   if ep.face_name == 'up']) < 4:

            yellow_corner_pieces = self.c.corner_pieces_matching_color(
                'yellow'
            )

            up_yellow_corner_pieces = [
                cp for cp in yellow_corner_pieces if cp.face_name == 'up'
            ]

            n_up_yellow_corners = len(up_yellow_corner_pieces)

            other_yellow_corner_pieces = [
                cp for cp in yellow_corner_pieces
                if cp not in up_yellow_corner_pieces
            ]

            other_yellow_corner_orientations = [
                self.non_up_yellow_corner_orientation(cp) for cp
                in other_yellow_corner_pieces
            ]

            # State 1. We need to have a yellow piece on the up left edge. Find
            # a corner piece on the up edge that we can rotate to put yellow on
            # the left face. Do the rotation.
            if n_up_yellow_corners == 0:
                cp_to_rotate = [
                    cp for cp, orientation in
                    list(zip(other_yellow_corner_pieces,
                             other_yellow_corner_orientations))
                    if orientation > 0
                ]
                cp_to_rotate = cp_to_rotate[0]

                self.c.rotate_face_edge_to_edge(
                    'up', cp_to_rotate.face_name, 'left'
                )

            # State 2. We have exactly 1 yellow piece with its face on the up
            # face. It needs to be in the up,left,front position. Do the
            # rotation.
            if n_up_yellow_corners == 1:
                cp_to_rotate = [cp for cp in up_yellow_corner_pieces][0]

                adj_faces = set(
                    [cp.face_name for cp in cp_to_rotate.adj_pieces]
                )

                # Move desired piece to front,left,up position
                if adj_faces == set(['left', 'back']):
                    self.c.rotate_face_edge_to_edge('up', 'left', 'front')
                elif adj_faces == set(['back', 'right']):
                    self.c.rotate_face_edge_to_edge('up', 'back', 'front')
                elif adj_faces == set(['front', 'right']):
                    self.c.rotate_face_edge_to_edge('up', 'right', 'front')
                else:
                    pass

            # State 3. We need to have a yellow piece on the up front edge.
            # Find a corner piece on the up edge that we can rotate to put
            # yellow on the front face. Do the rotaiton.
            if n_up_yellow_corners >= 2:
                cp_to_rotate = [
                    cp for cp, orientation in
                    list(zip(other_yellow_corner_pieces,
                             other_yellow_corner_orientations))
                    if orientation < 0
                ]
                cp_to_rotate = cp_to_rotate[0]

                self.c.rotate_face_edge_to_edge(
                    'up', cp_to_rotate.face_name, 'front'
                )

            # In position. Apply Stage 6 algorithm.
            self.c.perform_move_list([
                'R', 'U', '-1R', 'U', 'R', '2U', '-1R'
            ])

    def flip_front_corners(self):
        self.c.perform_move_list([
            '-1R', 'F', '-1R', '2B', 'R', '-1F', '-1R', '2B', '2R', '-1U'
        ])

    def position_yellow_corners(self):
        yellow_corner_pieces = self.c.corner_pieces_matching_color('yellow')
        solved_yellow_corner_pieces = [
            cp for cp in yellow_corner_pieces if self.c.corner_piece_solved(cp)
        ]

        # After rotation, at least two corner pieces should be solved
        while len(solved_yellow_corner_pieces) < 2:
            self.c.rotate_up(1)
            solved_yellow_corner_pieces = [
                cp for cp in yellow_corner_pieces if self.c.corner_piece_solved(cp)
            ]

        while len(solved_yellow_corner_pieces) <= 2:

            if len(solved_yellow_corner_pieces) == 1:
                raise Exception(
                    "Only one of the up/yellow corners is solved. \n" +
                    "The permutation to correct this is not yet implemented! \n" +
                    "We shouldn't be reaching this state, so the solver" +
                    "cannot currently handle this."
                )

            if len(solved_yellow_corner_pieces) == 2:
                cp1, cp2 = solved_yellow_corner_pieces
                common_solved_edge = list(
                    cp1.adjacent_face_names().intersection(cp2.adjacent_face_names())
                )

                # If there's a common solved edge, rotate cube so it's on the back.
                # Then flip the front corners to solve the cube.
                if common_solved_edge:
                    common_solved_edge = common_solved_edge[0]
                    self.c.rotate_cube_face_to_face(common_solved_edge, 'back')
                    self.flip_front_corners()

                # no common edge, so solved pieces are diagonal to one another
                else:
                    # If neither of the pieces is on the front/right/up
                    # position, rotate the cube once to get the solved diagonal
                    # pieces to be on front/right/up and back/left/up
                    if (cp1.adjacent_face_names() != set(['front', 'right']) and
                            cp2.adjacent_face_names() != set(['front', 'right'])):
                        self.c.rotate_cube_face_to_face('right', 'front')

                    # We want to move the front/left/up piece to the
                    # back/right/up position to solve it
                    # We will do this by
                    #  1) flipping front corners (this moves it to
                    #     front/right/up in the starting orientation)
                    #  2) rotating the cube CW once
                    #  3) flipping front corners (this moves it to
                    #     front/right/back in the starting orientaion)
                    #  4) rotating the cube CCW once (returns us to
                    #     our original orientation)
                    self.flip_front_corners()
                    self.c.rotate_cube_face_to_face('right', 'front')
                    self.flip_front_corners()
                    self.c.rotate_cube_face_to_face('front', 'right')

            solved_yellow_corner_pieces = [
                cp for cp in yellow_corner_pieces if self.c.corner_piece_solved(cp)]

    def permute_up_edge_pieces(self, orientation):
        if orientation == 'CW':
            self.c.perform_move_list([
                '2F', 'U', 'L', '-1R', '2F', '-1L', 'R', 'U', '2F'
            ])

        elif orientation == 'CCW':
            self.c.perform_move_list([
                '2F', '-1U', 'L', '-1R', '2F', '-1L', 'R', '-1U', '2F'
            ])

    def solve_yellow_edges(self):
        yellow_edge_pieces = self.c.edge_pieces_matching_color('yellow')
        solved_yellow_edge_pieces = [
            ep for ep in yellow_edge_pieces if self.c.edge_piece_solved(ep)]
        unsolved_yellow_edge_pieces = [
            ep for ep in yellow_edge_pieces if not self.c.edge_piece_solved(ep)]

        while len(solved_yellow_edge_pieces) == 0:
            self.permute_up_edge_pieces('CW')
            solved_yellow_edge_pieces = [
                ep for ep in yellow_edge_pieces if self.c.edge_piece_solved(ep)]
            unsolved_yellow_edge_pieces = [
                ep for ep in yellow_edge_pieces if not self.c.edge_piece_solved(ep)]

        if len(solved_yellow_edge_pieces) == 1:
            # Identify the solved edge / face and rotate it to the back
            solved_edge_piece = solved_yellow_edge_pieces[0]
            solved_face_name = list(solved_edge_piece.adjacent_face_names())[0]
            self.c.rotate_cube_face_to_face(solved_face_name, 'back')

            # List out all of the adjacent piece's unsolved face names and
            # their colors, along with the opposite faces and their colors
            unsolved_face_names = [list(ep.adjacent_face_names())[
                0] for ep in unsolved_yellow_edge_pieces]
            unsolved_edge_colors = [
                cp.adj_pieces[0].color_name for cp in unsolved_yellow_edge_pieces
            ]
            unsolved_opposite_face_names = [
                opposite_face[f] for f in unsolved_face_names
            ]
            unsolved_opposite_face_colors = [
                self.c.face_color(f) for f in unsolved_opposite_face_names
            ]

            # One of the unsovled pieces is opposite to the face it needs to go
            # to. Identify this piece's index in the unsolved edge list
            idx = [
                i for i, pair in
                enumerate(zip(unsolved_edge_colors, unsolved_opposite_face_colors))
                if pair[0] == pair[1]
            ]
            idx = idx[0]

            # If the unsolved adjacent piece is on the right side, we need to
            # perform a CCW permutation to move it to the left side and solve
            # the cube
            if unsolved_face_names[idx] == 'right':
                self.permute_up_edge_pieces('CCW')

            # If the unsolved adjacent piece is on the left side, we need to
            # perform a CW permutation to move it to the right side and solve
            # the cube
            elif unsolved_face_names[idx] == 'left':
                self.permute_up_edge_pieces('CW')

    def solve(self):
        self.initial_state = self.c.export_state()
        self.c.flush_recorder()
        self.c.start_recorder()
        self.solve_white_cross()
        self.solve_white_corners()
        self.solve_middle_edges()
        self.solve_yellow_cross()
        self.solve_yellow_corners()
        self.position_yellow_corners()
        self.solve_yellow_edges()
        self.c.stop_recorder()
        self.solve_moves = self.c.recorded_moves[:]
        self.recorded_moves = self.solve_moves

    def solve_and_plot_after_steps(self):
        self.initial_state = self.c.export_state()
        self.c.flush_recorder()
        self.c.start_recorder()
        self.c.cube_plot(title_str='initial state')
        self.solve_white_cross()
        self.c.cube_plot(title_str='after white cross')
        self.solve_white_corners()
        self.c.cube_plot(title_str='after white corners')
        self.solve_middle_edges()
        self.c.cube_plot(title_str='after middle edges')
        self.solve_yellow_cross()
        self.c.cube_plot(title_str='after yellow cross')
        self.solve_yellow_corners()
        self.c.cube_plot(title_str='after yellow corners')
        self.position_yellow_corners()
        self.c.cube_plot(title_str='after positioning yellow corners')
        self.solve_yellow_edges()
        self.c.cube_plot(title_str='after permuting yellow edges')
        self.c.stop_recorder()
        self.solve_moves = self.c.recorded_moves[:]
        self.recorded_moves = self.solve_moves


if __name__ == '__main__':
    c = Cube()
    c.randomize_state()
    s = Solver(c)
    s.solve_and_plot_after_steps()
    print(c.recorded_moves)
    plt.show()
