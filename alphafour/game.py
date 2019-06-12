import numpy as np
import agent

class ConnectGame(object):
    """An instance of a Connect Four game. 

    Responsible for handling logic of player turns and end game results
    """    

    def __init__(self, player1: agent.Agent, player2: agent.Agent, move_delay: int=0) -> None:
        """Initializes a game instance.

        Args:
            player1 (Agent): An Agent instance. player1 will move first
            player2 (Agent): Another Agnet instance.
            move_delay (int): The minimum delay in seconds between each move. Can make the 
                game more watchable if both agents are fast algorithms. If an Agent takes
                longer than move_delay to finish their turn, the turn will advance as soon 
                as the Agent submits their move
        """

        self._player1 = player1
        self._player2 = player2
        self._move_delay = move_delay

        self._turn = 1
        self._game_state = np.zeros((10, 6, 7))  # 10, 6x7 boards to store the last 5 moves for each player
        self._game_board = np.zeros((6,7))  # A single, printable game board that humans can read


    def play_game(self) -> None:
        """Plays a game of Connect Four
        
        TODO: Uh. The whole thing
        """

        while not self.game_finished():
            print("AHHHHH")


    def game_finished(self) -> bool:
        """Checks the game_state to see if the game has finished
        
        TODO: All the logic for this
        """
        
        return False


    def validate_move(self, move) -> bool:
        """Validates whether or not the proposed move is valid

        TODO: All the logic for this
        """

        return True