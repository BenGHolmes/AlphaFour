import numpy as np
from agents import Agent
import time
import math
import helpers

class AlphaBeta(Agent):
    """Agent that implements minimax with alpha-beta pruning to select its next move."""

    def __init__(self, name: str = None) -> None:
        self._name = name

        # Initialize array of indices used by the static value function. This is kinda ugly,
        # but it's more than twice as fast as building them on the fly, so it's worth it.
        self._window_indices = np.array([
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



    def get_move(self, game_board: np.ndarray, agent_marker: int) -> np.ndarray:
        """Recursively runs minimax to determine the best move to make. 

        Recursively runs minimax algorithm with alpha-beta pruning starting at the current game state.
        This player is assumed to be maximizing.

        Args:
            game_board (np.ndarray): A human readable version of the board, with all
                currently played pieces represented as a 1 or 2 for players one and 
                two respectively. All open spaces are 0
            agent_marker (int): Integer indicating which value in game_board corresponds
                to this Agent's pieces

        Returns:
            An ndarray representing the move, with a 1 in the row,col of the new
            piece, and all other entries zero
        """

        start = time.time()
        move_val, move = self.alpha_beta(game_board, depth=5)
        end = time.time()

        print("Found optimal move with value: {}, in {}s".format(move_val, (end - start)))
        return move


    def alpha_beta(self, game_board: np.ndarray, alpha: float = -np.inf, beta: float = np.inf, 
        depth: int=np.inf, max_player: bool = True) -> (int, np.ndarray):
        """Perform minimax with alpha-beta pruning to determine best move to take from current game_board.

        Performs minimax starting at the current position and ending after looking depth moves ahead, or when all leaf
        nodes are end_game states.

        TODO: If multiple winning moves, it picks the first one. Change so agent chooses the quickest win

        Args:
            game_board (np.ndarray): 2D array representing the current pieces as 1 or -1 if they
                are for the maximizing or minimizing player respectively.
            alpha (float, optional): The best score achieved by the maximizing player. Defaults to -np.inf,
                the worst possible value for the maximizing player.
            beta (float, optional): The best score achieved by the minimizing player. Defaults to np.inf.
            depth (int, optional): The number of layers to check using minimax. Defualt is np.inf which will
                check all layers.
            max_player (bool, optional): Indicates whether the turn at the root node belongs to the minimizing or
                maximizing player. Default is True, meaning the maximizing player is next to move.

        Returns:
            move_val (int): The optimal value of this node.
            move (np.ndarray): A 6x7 numpy array with a 1 in the spot of the move to take from the current
                node that will result in the optimal value.
        """

        legal_moves = helpers.get_legal_moves(game_board)

        if legal_moves.size == 0 or depth == 0:
            # Leaf node, perform static value checking.
            return self.get_static_value(game_board), None

        next_states = game_board + legal_moves if max_player else game_board - legal_moves
        best_move = legal_moves[0]

        while next_states.size > 0:
            best_idx = self.get_most_valuable(next_states, max_player)
            state = next_states[best_idx]
            next_states = np.delete(next_states, best_idx, 0)

            # Only recurse farther if the current state is not an end game state
            if math.isinf(self.get_static_value(state)):
                val = self.get_static_value(state)
            else:
                val, _ = self.alpha_beta(state, alpha=alpha, beta=beta, depth=depth-1,max_player=not max_player)

            if max_player and val > alpha:
                alpha = val
                best_move = legal_moves[best_idx]
            elif not max_player and val < beta:
                best_move = legal_moves[best_idx]
                beta = val

            legal_moves = np.delete(legal_moves, best_idx, 0)

            if beta < alpha:
                break


        if max_player:
            return alpha, best_move
        else:
            return beta, best_move


    def get_most_valuable(self, states: np.ndarray, max_player: bool) -> int:
        """Return the index of next_states corresponding to the best static value for current player.

        Args:
            states (np.ndarray): Numpy array of 6x7 board states. Maximizing player is 1,minimizing
                player is -1.
            max_player (bool): If max_player is true, return the index with maximum static value,
                if false, return the index that minimizes static value.
        """

        idx = 0
        best_val = self.get_static_value(states[0])

        for i in range(1,states.shape[0]):
            val = self.get_static_value(states[i])

            if max_player and val > best_val:
                idx = i
                best_val = val
            elif val < best_val:
                idx = i
                best_val = val

        return idx

        
    def get_static_value(self, game_board: np.ndarray) -> float:
        """Returns the static value of game_board.

        For each possible way to get four in a row, check if the line contains only 1 or -1.
        If that row contains pieces from only one player, add the sum of their pieces to value.
        If either player has 4 in a row, return +/- inf

        Args:
            game_board (np.ndarray): The current minimax board with maximing player as 1
                and minimizing player as -1.

        Returns:
            value (float): The static value of the current position.
        """    
        windows = game_board.flatten()[self._window_indices].reshape(-1,4)
        uncontested_windows = windows[windows.min(axis=1) != -windows.max(axis=1)]
        if uncontested_windows.size == 0:
            return 0
        
        window_sums = uncontested_windows.sum(axis=1)
        return (abs(window_sums) * window_sums**2 / window_sums).sum()

        return value


    def handle_invalid_move(self) -> None:
        # Throw exception during development
        # TODO: Add some nice handler later on
        raise Exception
        