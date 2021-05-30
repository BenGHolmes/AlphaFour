import numpy as np
from agents import Agent


class Human(Agent):
    """Agent that prompts for user input each move. No independent decisions.

    Each turn the game board is printed, and the user is prompted to
    enter their next move using the keyboard.
    """
    def get_move(self, game_board: np.ndarray) -> np.ndarray:
        """Prompts the user for their next move.

        Prompts the user for what column to add the piece to, calculates the
        row the piece will land in, and submits the move the the ConnectGame.
        If the move is illegal, handle_invalid_move is called.

        Args:
            game_board (np.ndarray): A human readable version of the board, with all
                currently played pieces represented as a 1 or 2 for players one and
                two respectively. All open spaces are 0.

        Returns:
            An ndarray representing the move, with a 1 in the row,col of the new
            piece, and all other entries zero.
        """
        col_idx = None
        while col_idx is None:
            try:
                move_string = input("Enter move:".format())
                col_idx = int(move_string)
            except KeyboardInterrupt:
                exit()
            except:
                print("Invalid move: {}".format(move_string))
                pass

        new_state = self.get_new_state(col_idx, game_board)
        return new_state

    def get_new_state(self, col_idx: int, board: np.ndarray) -> np.ndarray:
        """Calculate the row the piece lands in and return the new game_state.

        Args:
            col_idx (int): The column the player chose for their move
                board (np.ndarray): The current board_state.

        Returns:
            An ndarray representing the move, with a 1 in the row,col of the new
            piece, and all other entries zero.
        """

        col = board[:, col_idx]

        filled = np.nonzero(col)[0]  # Returns tuple
        if filled.size > 0:
            row_idx = (
                min(filled) - 1
            )  # Place on top of the highest row with a non-zero value
        else:
            row_idx = 5  # If no pieces in that row, new piece goes at the bottom

        new_state = np.zeros((6, 7))
        new_state[row_idx, col_idx] = 1

        return new_state

    def handle_invalid_move(self) -> None:
        """Prints invalid move to the screen"""

        print("========================")
        print("Invalid move! Try again.")
        print("========================")
