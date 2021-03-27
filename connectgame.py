import numpy as np
from agents import Agent, Human

class ConnectGame(object):
    """An instance of a Connect Four game. 

    Responsible for handling logic of player turns and end game results
    """    

    def __init__(self, player1: Agent, player2: Agent, move_delay: int=3) -> None:
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

        self._turn = 0

        # Representation of the game board. Current player's positions are 1, opponent is -1
        self._game_board = np.zeros((6,7))  

        # Initialize array of indices used to check for winning groups. This is kinda ugly,
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


    def play_game(self) -> None:
        """Plays a game of Connect Four"""

        while not self.game_finished():
            self.print_board()

            if self._turn % 2 == 0:
                curr_player = self._player1
            else:
                curr_player = self._player2

            move = curr_player.get_move(self._game_board, self._turn % 2 + 1)

            if self.validate_move(move):
                self._game_board += move  # Add a 1 for current player to the game board
                self._game_board *= -1    # Flip signs to represent current state
                self._turn += 1           # Incremenet turn
            else:
                curr_player.handle_invalid_move()

        self.print_board()

        if self._winner is not None:
            # TODO: Cooler artwork for the big winner
            print("======================")
            print(self._winner._name, "WINS!")
            print("======================")

        else:
            print("======================")
            print("TIE GAME!")
            print("======================")



    def game_finished(self) -> bool:
        """Checks the game_state to see if the game has finished.
        
        Since this check comes before each turn, we only need to check if the player who
        played last (meaning their pieces have value -1) won on the last turn.
        """

        windows = self._game_board.flatten()[self._window_indices].reshape(-1,4)
        uncontested_windows = windows[windows.min(axis=1) != -windows.max(axis=1)]
        if uncontested_windows.size > 0:
            min_sum = uncontested_windows.sum(axis=1).min()

            if min_sum == -4:
                if self._turn%2 == 0:
                    # If it is now player1's turn, then player2 won last turn
                    self._winner = self._player2
                else:
                    # Otherwise player1 won last turn
                    self._winner = self._player1

                return True

        if 0 not in self._game_board:
            self._winner = None
            return True
        
        return False


    def validate_move(self, move: np.ndarray) -> bool:
        """Validates whether or not the proposed move is valid.

        Args:
            move (np.ndarray): The proposed move, with a one in the row,col of
            the new piece, and zeros in all other squares.

        Returns:
            is_valid (bool): True if the proposed move is valid, else False
        """

        # Return False if there is not a single entry with value 1
        if move.sum() != 1 or move.max() != 1:
            return False

        # Get row and column of new move
        mov_idx = np.argmax(move)
        row = int(mov_idx / 7)
        col = int(mov_idx % 7) if row else mov_idx

        # Move is valid if that square is open, and either row==5 (bottom) or 
        # the square below is occupied
        is_empty = self._game_board[row, col] == 0
        valid_height = ((row == 5) or self._game_board[row+1, col] != 0)
        
        return (is_empty and valid_height)


    def print_board(self) -> None:
        """Prints the game board to the console.

        Prints the current game board with player one as X and player two as O.
        """

        print("\n\n===============\n\n")

        if self._turn % 2 == 0:
            # P1 is 1, P2 is -1
            current_marker = 'X'
            opponent_marker = 'O'
        else:
            # P2 is 1, P1 is -1
            current_marker = 'O'
            opponent_marker = 'X'

        for row in self._game_board:
            row_str = [current_marker if x == 1 else  opponent_marker if x == -1 else '_' for x in row]
            print('|' + '|'.join(row_str) + '|')

        print('|0|1|2|3|4|5|6|')
    
        print('\n{}: X, {}: O'.format(self._player1._name, self._player2._name))


