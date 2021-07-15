import numpy as np


class Agent(object):
    """Generic Agent class. To be used as a parent class for different implementations."""

    def __init__(self) -> None:
        pass

    def get_move(self, game_board: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def handle_invalid_move(self) -> None:
        raise NotImplementedError
