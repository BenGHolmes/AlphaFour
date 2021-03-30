import numpy as np

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

def get_legal_moves(game_board: np.ndarray) -> np.ndarray:
    """Get all possible legal moves from current game board.

    Args:
        game_board (np.ndarray): The current board with current player as 1 and 
            opponent player as -1.
    
    Returns:
        legal_moves (np.ndarray): Numpy array of all legal moves that can be taken from current
            game_board. Each move array has the location of the new piece as a 1, with all 
            other squares as 0.
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
    
def winner(game_board):
    """Returns the winner/value of the game board.

    For each possible way to get four in a row, check if the line sums to 4 or -4
    and return the winner as 1 or -1. If no spaces remain, return 0 for a tie. If no
    one has won and moves can still be made, return None

    Args:
        game_board (np.ndarray): The current minimax board with maximing player as 1
            and minimizing player as -1.

    Returns:
        1 if current player has won, -1 if opponent has won, 0 for a tie, None for
        a state that doesn't end the game.
    """    
    windows = game_board.flatten()[WINDOW_INDICES].reshape(-1,4)
    uncontested_windows = windows[windows.min(axis=1) != -windows.max(axis=1)]
    
    # If there are any windows with only 1's or -1's, check if any are full
    if uncontested_windows.size > 0:
        window_sums = uncontested_windows.sum(axis=1)
        if window_sums.max() == 4:
            return 1
        elif window_sums.min() == -4:
            return -1
    # If no zeros on board, game ended in tie
    elif not (game_board == 0).any():
        return 0 
    
    return None