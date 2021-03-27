class MCTS(Agent):
    """Agent that implements Monte Carlo Tree Search to select next move."""

    def __init__(self, name: str = None) -> None:
        self._name = name

    def get_move(self, game_board: np.ndarray, agent_marker: int) -> np.ndarray:

        
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


    def handle_invalid_move(self) -> None:
        # Throw exception during development
        # TODO: Add some nice handler later on
        raise Exception
        

