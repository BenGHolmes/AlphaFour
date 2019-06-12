import numpy as np
from connectgame import ConnectGame

class Agent(object):
    """Generic Agent class. To be used as a parent class for different implementations"""
    
    def __init__(self, name: str = None) -> None:
        self._name = name


    def get_move(self, game: ConnectGame) -> np.ndarray:
        """Returns the Agent's next move, based on the game_state
        
        Args:
            game (ConnectGame): the ConnectGame instance this Agent is playing in
        """

        pass


    def handle_invalid_move(self) -> None:
        """Called when the ConnectGame rejects the proposed move"""
        
        pass


class Human(Agent):
    """Agent that prompts for user input each move. No independent decisions.

    Each turn the game board is printed, and the user is prompted to
    enter their next move using the keyboard.
    """

    def __init__(self, name: str = None) -> None:
        self._name = name


    def get_move(self, game: ConnectGame) -> np.ndarray:
        """Prompts the user for their next move.

        Prompts the user for what column to add the piece to, calculates the 
        row the piece will land in, and submits the move the the ConnectGame. 
        If the move is illegal, handle_invalid_move is called.

        Args:
            game (ConnectGame): An instance of the current game.

        Returns:
            An ndarray representing the move, with a 1 in the row,col of the new
            piece, and all other entries zero
        """

        col_idx = input("{self._name}'s move:")

        new_state = self.get_new_state(col_idx, game._game_board)
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
        row_idx = min(np.nonzero(col)) - 1  # Place on top of the highest row with a non-zero value

        new_state = np.zeros((6,7))
        new_state[row_idx, col_idx] = 1

        return new_state

        