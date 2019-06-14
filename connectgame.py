import numpy as np
from agents import Agent, Human

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

        self._turn = 0
        self._game_state = np.zeros((10, 6, 7))  # 10, 6x7 boards to store the last 5 moves for each player
        self._game_board = np.zeros((6,7))  # A single, printable game board that humans can read


    def play_game(self) -> None:
        """Plays a game of Connect Four"""

        while not self.game_finished():
            self.print_board()

            if self._turn % 2 == 0:
                curr_player = self._player1
            else:
                curr_player = self._player2

            move = curr_player.get_move(self._game_state, self._game_board, self._turn % 2 + 1)

            if self.validate_move(move):
                self.commit_move(move)
                self._turn += 1
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
        
        TODO: Find something more efficient, or at the very least more interesting
        """

        # Define flattened array equivalents of horizontal, front slash (/) and back
        # slash (\) lines of 4 in the 2D array relative to starting point 0.
        search_arr = self._game_board.flatten()
        vertical_window = np.array([0,7,14,21])  # 0 is top point
        horizontal_window = np.array([0,1,2,3])  # 0 is left most point
        f_slash_window = np.array([0,6,12,18])  # 0 is top right point
        b_slash_window = np.array([0,8,16,24])  # 0 is top left point

        winner = 0

        # Check for vertical wins. Top piece must be in row [0,1,2] and any col [0..6]. In the flattened
        # array, that corresponds to indices [0:20] inclusive.
        for start in range(21):
            window = search_arr[vertical_window + start]
            if window.max() == window.min() and window.min() != 0:
                winner = window[0]
                break

        # Check for forward diagonal (/) wins. Top right piece must be in row [0,1,2] and col [3..6].
        if winner == 0:
            for start in [col + 7*row for col in range(3,7) for row in range(3)]:
                window = search_arr[f_slash_window + start]
                if window.max() == window.min() and window.min() != 0:
                    winner = window[0]
                    break

        # Check for back diagonal (\) wins. Top left piece must be in row [0,1,2] and col [0..3].
        if winner == 0:
            for start in [col + 7*row for col in range(4) for row in range(3)]:
                window = search_arr[b_slash_window + start]
                if window.max() == window.min() and window.min() != 0:
                    winner = window[0]
                    break

        # Check for horizontal wins. Left most piece must be in row [0..5] and col [0..3].
        if winner == 0:
            for start in [col + 7*row for col in range(4) for row in range(6)]:
                window = search_arr[horizontal_window + start]
                if window.max() == window.min() and window.min() != 0:
                    winner = window[0]
                    break

        if winner == 1:
            self._winner = self._player1
            return True

        if winner == 2:
            self._winner = self._player2
            return True

        # If no more spaces, game is over but there is no winner
        if 0 not in search_arr:
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

        print("\n\n===============\n\n")

        for row in self._game_board:
            row_str = ['X' if x == 1 else 'O' if x == 2 else '_' for x in row]
            print('|' + '|'.join(row_str) + '|')

        print('|0|1|2|3|4|5|6|')
    
        print('\n{}: X, {}: O'.format(self._player1._name, self._player2._name))


