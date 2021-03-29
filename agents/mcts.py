class MCTS(Agent):
    """Agent that implements Monte Carlo Tree Search to select next move."""

    def __init__(self, name: str = None) -> None:
        self._name = name


    def get_move(self, game_board: np.ndarray, agent_marker: int) -> np.ndarray:
        # TODO: This
        return
        

    def get_static_value(self, minimax_board: np.ndarray) -> float:
        """Returns the static value of game_board.

        For each possible way to get four in a row, check if the line contains only 1 or -1.
        If that row contains pieces from only one player, add the sum of their pieces to value.
        If either player has 4 in a row, return +/- inf.

        TODO: See if this is best? Maybe MCTS shold just use a 1 for win, 0 for tie, -1 for loss.

        Args:
            game_board (np.ndarray): The current minimax board with maximing player as 1
                and minimizing player as -1.

        Returns:
            value (float): The static value of the current position.
        """    
        windows = game_board.flatten()[helpers.WINDOW_INDICES].reshape(-1,4)
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
        

