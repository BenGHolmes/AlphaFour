import numpy as np
from agents import Agent

class Human(Agent):
    """Agent that prompts for user input each move. No independent decisions.

    Each turn the game board is printed, and the user is prompted to
    enter their next move using the keyboard.
    """

    def __init__(self, name: str = None) -> None:
        self._name = name


    def get_move(self, game_state: np.ndarray, game_board: np.ndarray, agent_marker: int) -> np.ndarray:
        """Prompts the user for their next move.

        Prompts the user for what column to add the piece to, calculates the 
        row the piece will land in, and submits the move the the ConnectGame. 
        If the move is illegal, handle_invalid_move is called.

        Args:
            game_state (np.ndarray): Current game state. A stack of 10 6x7 arrays 
                representing the last 5 moves for each player. layer 0-4 are player
                one's moves, and 5-9 are player two's moves. A 1 indicates where the
                new piece was played, and all other entries are 0
            game_board (np.ndarray): A human readable version of the board, with all
                currently played pieces represented as a 1 or 2 for players one and 
                two respectively. All open spaces are 0
            agent_marker (int): Integer indicating which value in game_board corresponds
                to this Agent's pieces

        Returns:
            An ndarray representing the move, with a 1 in the row,col of the new
            piece, and all other entries zero
        """

        col_idx = int(input("{}'s move:".format(self._name)))

        new_state = self.get_new_state(col_idx, game_board)
        return new_state


    def get_new_state(self, col_idx: int, board: np.ndarray) -> np.ndarray:
        """Calculate the row the piece lands in and return the new game_state

        Args:
            col_idx (int): The column the player chose for their move
            board (np.ndarray): The current board_state

        Returns:
            An ndarray representing the move, with a 1 in the row,col of the new
            piece, and all other entries zero
        """

        col = board[:,col_idx]

        filled = np.nonzero(col)[0]  # Returns tuple
        if filled.size > 0:
            row_idx = min(filled) - 1  # Place on top of the highest row with a non-zero value
        else:
            row_idx = 5  # If no pieces in that row, new piece goes at the bottom

        new_state = np.zeros((6,7))
        new_state[row_idx, col_idx] = 1

        return new_state


    def handle_invalid_move(self) -> None:
        """Prints invalid move to the screen"""
        
        print("========================")
        print("Invalid move! Try again.")
        print("========================")

