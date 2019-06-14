import numpy as np

class Agent(object):
    """Generic Agent class. To be used as a parent class for different implementations"""
    
    def __init__(self, name: str = None) -> None:
        self._name = name


    def get_move(self, game_state: np.ndarray, game_board: np.ndarray) -> np.ndarray:
        """Returns the Agent's next move, based on the game_state
        
        Args:
            game_state (np.ndarray): Current game state. A stack of 10 6x7 arrays 
                representing the last 5 moves for each player. layer 0-4 are player
                one's moves, and 5-9 are player two's moves. A 1 indicates where the
                new piece was played, and all other entries are 0
            game_board (np.ndarray): A human readable version of the board, with all
                currently played pieces represented as a 1 or 2 for players one and 
                two respectively. All open spaces are 0
        """

        pass


    def handle_invalid_move(self) -> None:
        """Called when the ConnectGame rejects the proposed move"""
        
        pass

