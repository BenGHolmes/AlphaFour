import numpy as np

def get_legal_moves(game_board: np.ndarray) -> np.ndarray:
    """Get all possible legal moves from current minimax_board.

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