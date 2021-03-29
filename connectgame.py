import numpy as np
from agents import Agent, Human
import helpers

class ConnectGame(object):
    """An instance of a Connect Four game. 

    Responsible for handling logic of player turns and end game results.
    """    

    def __init__(self, player1: Agent, player2: Agent) -> None:
        """Initializes a game instance.

        Args:
            player1 (Agent): An Agent instance. player1 will move first.
            player2 (Agent): Another Agnet instance.
        """

        self._player1 = player1
        self._player2 = player2

        self._turn = 0

        # Representation of the game board. Current player's positions are 1, opponent is -1
        self._game_board = np.zeros((6,7))  


    def play_game(self) -> None:
        """Plays a game of Connect Four."""

        while not self.game_finished():
            self.print_board()

            if self._turn % 2 == 0:
                curr_player = self._player1
            else:
                curr_player = self._player2

            move = curr_player.get_move(self._game_board)

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

        windows = self._game_board.flatten()[helpers.WINDOW_INDICES].reshape(-1,4)
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
            is_valid (bool): True if the proposed move is valid, else False.
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


