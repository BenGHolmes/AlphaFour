import numpy as np
from agents import Agent
import time
import math

class AlphaBeta(Agent):
    """Agent that implements minimax with alpha-beta pruning to select its next move."""

    def __init__(self, name: str = None) -> None:
        self._name = name


    def get_move(self, game_state: np.ndarray, game_board: np.ndarray, agent_marker: int) -> np.ndarray:
        """Recursively runs minimax to determine the best move to make. 

        Recursively runs minimax algorithm with alpha-beta pruning starting at the current game state.
        This player is assumed to be maximizing.

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

        minimax_board = self.get_minimax_board(game_board, agent_marker)

        start = time.time()
        move_val, move = self.alpha_beta(minimax_board, depth=5)
        end = time.time()

        print("Found optimal move with value: {}, in {}s".format(move_val, (end - start)))
        return move


    def alpha_beta(self, minimax_board: np.ndarray, alpha: float = -np.inf, beta: float = np.inf, 
        depth: int=np.inf, max_player: bool = True) -> (int, np.ndarray):
        """Perform minimax with alpha-beta pruning to determine best move to take from current minimax_board.

        Performs minimax starting at the current position and ending after looking depth moves ahead, or when all leaf
        nodes are end_game states.

        TODO: If multiple winning moves, it picks the first one. Change so agent chooses the quickest win

        Args:
            minimax_board (np.ndarray): 2D array representing the current pieces as 1 or -1 if they
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

        legal_moves = self.get_legal_moves(minimax_board)

        if legal_moves.size == 0 or depth == 0:
            # Leaf node, perform static value checking.
            return self.get_static_value(minimax_board), None

        next_states = minimax_board + legal_moves if max_player else minimax_board - legal_moves
        best_move = legal_moves[0]

        while next_states.size > 0:
            best_idx = self.get_most_valuable(next_states, max_player)
            state = next_states[best_idx]
            next_states = np.delete(next_states, best_idx, 0)

            # Only recurse farther if the current state is not an end game state
            if math.isinf(self.get_static_value(state)):
                val = self.get_static_value(state)
            else:
                val, _ = self.alpha_beta(state, alpha=alpha, beta=beta, depth=depth-1, max_player=not max_player)

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
            states (np.ndarray): Numpy array of 6x7 board states. Maximizing player is 1, minimizing
                player is -1.
            max_player (bool): If max_player is true, return the index with maximum static value,
                if false, return the index that minimizes static value.
        """

        idx = 0
        best_val = self.get_static_value(states[0])

        for i in range(1, states.shape[0]):
            val = self.get_static_value(states[i])

            if max_player and val > best_val:
                idx = i
                best_val = val
            elif val < best_val:
                idx = i
                best_val = val

        return idx

        
    def get_static_value(self, minimax_board: np.ndarray) -> float:
        """Returns the static value of minimax_board.

        For each possible way to get four in a row, check if the line contains only 1 or -1.
        If that row contains pieces from only one player, add the sum of their pieces to value.
        If either player has 4 in a row, return +/- inf

        Args:
            minimax_board (np.ndarray): The current minimax board with maximing player as 1
                and minimizing player as -1.

        Returns:
            value (float): The static value of the current position.
        """

        value = 0

        # Search windows for each possible type of four in a row in 2D
        search_arr = minimax_board.flatten()
        vertical_window = np.array([0,7,14,21])  # 0 is top point
        horizontal_window = np.array([0,1,2,3])  # 0 is left most point
        f_slash_window = np.array([0,6,12,18])  # 0 is top right point
        b_slash_window = np.array([0,8,16,24])  # 0 is top left point

        # Check for vertical wins. Top piece must be in row [0,1,2] and any col [0..6]. In the flattened
        # array, that corresponds to indices [0:20] inclusive.
        for start in range(21):
            window = search_arr[vertical_window + start]
            if not (1 in window and -1 in window):
                win_val = window.sum()
                if abs(win_val) == 4: 
                    return np.inf * win_val
                value += win_val

        # Check for forward diagonal (/) wins. Top right piece must be in row [0,1,2] and col [3..6].
        for start in [col + 7*row for col in range(3,7) for row in range(3)]:
            window = search_arr[f_slash_window + start]
            if not (1 in window and -1 in window):
                win_val = window.sum()
                if abs(win_val) == 4: 
                    return np.inf * win_val
                value += win_val

        # Check for back diagonal (\) wins. Top left piece must be in row [0,1,2] and col [0..3].
        for start in [col + 7*row for col in range(4) for row in range(3)]:
            window = search_arr[b_slash_window + start]
            if not (1 in window and -1 in window):
                win_val = window.sum()
                if abs(win_val) == 4: 
                    return np.inf * win_val
                value += win_val

        # Check for horizontal wins. Left most piece must be in row [0..5] and col [0..3].
        for start in [col + 7*row for col in range(4) for row in range(6)]:
            window = search_arr[horizontal_window + start]
            if not (1 in window and -1 in window):
                win_val = window.sum()
                if abs(win_val) == 4: 
                    return np.inf * win_val
                value += win_val

        return value


    def get_legal_moves(self, minimax_board: np.ndarray) -> np.ndarray:
        """Get all possible legal moves from current minimax_board.

        Args:
            minimax_board (np.ndarray): The current board with maximizing player as 1 and 
                minimizing player as -1.
        
        Returns:
            legal_moves (np.ndarray): Numpy array of all legal moves that can be taken from current
                minimax_board. Each move array has the location of the new piece as a 1, with all 
                other squares as 0.
        """

        legal_moves = np.ndarray(0)

        for col_idx in range(7):
            col = minimax_board[:,col_idx]
            if (col == 0).sum() > 0:
                row_idx = int(np.argwhere(col==0).max())
                move = np.zeros((6,7))
                move[row_idx, col_idx] = 1
                legal_moves = np.append(move, legal_moves)

        legal_moves = legal_moves.reshape(-1,6,7)

        return legal_moves


    def get_minimax_board(self, game_board: np.ndarray, agent_marker:int) -> np.ndarray:
        """Returns game_board with this Agent's pieces as 1 and opponent's pieces as -1

        Args:
            game_board (np.ndarray): Current game board with player one's pieces as 1 and 
                player two's pieces as 2.
            agent_marker (int): This Agent's piece value. Either 1 or 2.

        Returns:
            minimax_board (np.ndarray): game_board with this Agent's pieces as 1 and 
                opponent's pieces as -1 
        """

        minimax_board = np.zeros((6,7))
        minimax_board[game_board == agent_marker] = 1
        minimax_board[(game_board != agent_marker) & (game_board != 0)] = -1

        return minimax_board


    def handle_invalid_move(self) -> None:
        # Throw exception during development
        # TODO: Add some nice handler later on
        raise Exception
        