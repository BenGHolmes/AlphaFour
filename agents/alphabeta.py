import numpy as np
from agents import Agent
from connectboard import ConnectBoard
import time
import math

class AlphaBeta(Agent):
    """Agent that implements minimax with alpha-beta pruning to select its next move."""

    def get_move(self, game_board: np.ndarray) -> np.ndarray:
        """Recursively runs minimax to determine the best move to make. 

        Recursively runs minimax algorithm with alpha-beta pruning starting at the current game state.
        This player is assumed to be maximizing.

        Args:
            game_board (np.ndarray): current board with a 1 for current player, -1 for
                opponent, and 0 for open space

        Returns:
            An ndarray representing the move, with a 1 in the row,col of the new
            piece, and all other entries zero.
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
        legal_moves = ConnectBoard.get_legal_moves(game_board)

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
        If either player has 4 in a row, return +/- inf.

        Args:
            game_board (np.ndarray): The current minimax board with maximing player as 1
                and minimizing player as -1.

        Returns:
            value (float): The static value of the current position.
        """    
        windows = game_board.flatten()[ConnectBoard.WINDOW_INDICES].reshape(-1,4)
        uncontested_windows = windows[windows.min(axis=1) != -windows.max(axis=1)]
        if uncontested_windows.size == 0:
            return 0
        
        window_sums = uncontested_windows.sum(axis=1)

        if window_sums.max() == 4:
            return np.inf
        elif window_sums.min() == -4:
            return -np.inf
        else:
            return (abs(window_sums) * window_sums**2 / window_sums).sum()


    def handle_invalid_move(self) -> None:
        # Throw exception during development
        # TODO: Add some nice handler later on
        raise Exception
        