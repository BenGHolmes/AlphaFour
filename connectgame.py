import numpy as np
from agent import Agent, Human

class ConnectGame(object):
    """An instance of a Connect Four game. 

    Responsible for handling logic of player turns and end game results
    """    

    def __init__(self, player1: Agent, player2: Agent, move_delay: int=0) -> None:
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
        """Plays a game of Connect Four"""

        while not self.game_finished():
            self.print_board()

            if self._turn % 2 == 1:
                curr_player = self._player1
            else:
                curr_player = self._player2

            move = curr_player.get_move()

            if self.validate_move(move):
                self.commit_move(move)
                self._turn += 1
            else:
                curr_player.handle_invalid_move()


    def game_finished(self) -> bool:
        """Checks the game_state to see if the game has finished.
        
        TODO: All the logic for this
        """
        
        return False


    def validate_move(self, move: np.ndarray) -> bool:
        """Validates whether or not the proposed move is valid.

        TODO: All the logic for this
        Args:
            move (np.ndarray): The proposed move, with a one in the row,col of
            the new piece, and zeros in all other squares.
        """

        return True


    def commit_move(self, move: np.ndarray) -> None:
        """Adds the validated move to the game_state and game_board.
        
        Pushes the current move to the top of the players recent move stack, and adds
        either a 1 or a 2 to the game_board in the appropriate spot.

        Args:
            move (np.ndarray): The new, validated move. 1 at the row,col of the new piece, and
                0 elsewhere.
        """

        player_num = (self._turn % 2) + 1  # Value added to game_board

        self._game_board[move==1] = player_num

        if player_num == 1:
            # Delete the oldest history of player one's moves and inset new one at position 0
            self._game_state = np.delete(self._game_state, 4, 0)  
            self._game_state = np.insert(self._game_state, 0, move, 0)
        else:
            self._game_state = np.delete(self._game_state, 9, 0)
            self._game_state = np.insert(self._game_state, 5, move, 0)


    def print_board(self) -> None:
        """Prints the game board to the console.

        Prints the current game board with player one as X and player two as O.
        """

        for row in self._game_board:
            row_str = ['X' if x == 1 else 'O' if x == 2 else '_' for x in row]
            print('|', '|'.join(row_str), '|')

        print('|', '|'.join(range(7)), '|')
    
        print('\n\n{self._player1._name}: X, {self._player2._name}: O')


