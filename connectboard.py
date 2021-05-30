import numpy as np
from agents import Agent, Human

class InvalidMoveException(Exception):
    pass

class ConnectBoard(object):
    """An instance of a Connect Four game board. 

    Responsible for handling all logic associated with the board, such as listing 
    legal moves, determining the winner, and handling player moves.
    """

    # Array of indices of all possible 4-in-a-row combinations. This is kinda ugly,
    # but it's much much faster than building them on the fly, so it's worth it.
    WINDOW_INDICES = np.array([
        # Horizontal groups of 4
        0,1,2,3,       1,2,3,4,       2,3,4,5,       3,4,5,6,     # Row 1
        7,8,9,10,      8,9,10,11,     9,10,11,12,    10,11,12,13, # Row 2
        14,15,16,17,   15,16,17,18,   16,17,18,19,   17,18,19,20, # Row 3
        21,22,23,24,   22,23,24,25,   23,24,25,26,   24,25,26,27, # Row 4
        28,29,30,31,   29,30,31,32,   30,31,32,33,   31,32,33,34, # Row 5
        35,36,37,38,   36,37,38,39,   37,38,39,40,   38,39,40,41, # Row 6
        
        # Vertical groups of 4
        0,7,14,21,     1,8,15,22,     2,9,16,23,     3,10,17,24,    4,11,18,25,    5,12,19,26,    6,13,20,27,  # Row 1-4
        7,14,21,28,    8,15,22,29,    9,16,23,30,    10,17,24,31,   11,18,25,32,   12,19,26,33,   13,20,27,34, # Row 2-5
        14,21,28,35,   15,22,29,36,   16,23,30,37,   17,24,31,38,   18,25,32,39,   19,26,33,40,   20,27,34,41, # Row 3-6
        
        # Diagonal up right
        21,15,9,3,     22,16,10,4,    23,17,11,5,    24,18,12,6,  # Row 1-4
        28,22,16,10,   29,23,17,11,   30,24,18,12,   31,25,19,13, # Row 2-5
        35,29,23,17,   36,30,24,18,   37,31,25,19,   38,32,26,20, # Row 3-6
        
        # Diagonal down right
        0,8,16,24,     1,9,17,25,     2,10,18,26,    3,11,19,27,  # Row 1-4
        7,15,23,31,    8,16,24,32,    9,17,25,33,    10,18,26,34, # Row 2-5
        14,22,30,38,   15,23,31,39,   16,24,32,40,   17,25,33,41  # Row 3-6
    ])


    def __init__(self) -> None:
        """Initializes a game instance."""

        # Store game state. Board stores 1 for player1 and -1 for player 2
        self._game_board = np.zeros((6,7))  


    
    def current_state(self) -> np.ndarray:
        """Returns the current game board."""
        return self._game_board


    def make_move(self, move: np.ndarray):
        """Adds the given move to the current game board."""
        if not self._validate_move(abs(move)):
            raise InvalidMoveException
        
        self._game_board += move


    def winner(self) -> int:
        """Returns the winner/value of the game board.

        For each possible way to get four in a row, check if the line sums to 4 or -4
        and return the winner as 1 or -1. If no spaces remain, return 0 for a tie. If no
        one has won and moves can still be made, return None

        Returns:
            1 if player1 has won, 2 if player2 has won, 0 for a tie, and None for
            a state that doesn't end the game.
        """    
        windows = self._game_board.flatten()[self.WINDOW_INDICES].reshape(-1,4)
        uncontested_windows = windows[windows.min(axis=1) != -windows.max(axis=1)]
        
        # If there are any windows with only 1's or -1's, check if any are full
        if uncontested_windows.size > 0:
            window_sums = uncontested_windows.sum(axis=1)
            if window_sums.max() == 4:
                return 1
            elif window_sums.min() == -4:
                return 2
        # If no zeros on board, game ended in tie
        elif not (self._game_board == 0).any():
            return 0 
        
        return None


    def _validate_move(self, move: np.ndarray) -> bool:
        """Validates whether or not the proposed move is valid.

        Args:
            move (np.ndarray): The proposed move, with a one in the row,col of
                the new piece, and zeros in all other squares.

        Returns:
            True if the proposed move is valid, else False.
        """

        # Return False if there is not a single entry with value 1
        if move.sum() != 1 or move.max() != 1:
            return False

        # Get row and column of new move
        mov_idx = np.argmax(move)
        row = int(mov_idx / 7)
        col = int(mov_idx % 7) if row else mov_idx

        # Move is valid if that square is open, and either row==5 (bottom) or 
        # the square below is occupied
        is_empty = self._game_board[row, col] == 0
        valid_height = ((row == 5) or self._game_board[row+1, col] != 0)
        
        return (is_empty and valid_height)


    def __str__(self) -> str:
        """Prints the game board to the console.

        Prints the current game board with player one as X and player two as O.
        """
        board_string = ("\n\n===============\n\n")

        for row in self._game_board:
            row_str = ['X' if x == 1 else 'O' if x == -1 else '_' for x in row]
            board_string += ('|' + '|'.join(row_str) + '|\n')

        board_string += ('|0|1|2|3|4|5|6|\n')
    
        board_string += ('\nP1: X, P2: O\n')

        return board_string


    @staticmethod
    def get_legal_moves(game_board: np.ndarray) -> np.ndarray:
        """Get all possible legal moves from the given game board.
        
        Args:
            game_board (np.ndarray) - A ConnectFour game board with all occupied
                spaces non-zero and all empty spaces as 0.

        Returns:
            Numpy array of all legal moves that can be taken from current
                game_board. Each move is a 2D game board with the location 
                of the new piece as a 1, and all other squares as 0.
        """
        legal_moves = np.ndarray(0)

        rows = (5 - abs(game_board).sum(axis=0)).astype(int)
        for col in range(7):
            if rows[col] >= 0:
                move = np.zeros((6,7))
                move[rows[col], col] = 1
                legal_moves = np.append(move, legal_moves)

        legal_moves = legal_moves.reshape((-1,6,7))

        return legal_moves