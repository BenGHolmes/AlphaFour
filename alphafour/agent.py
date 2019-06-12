import numpy as np
import game

class Agent(object):
    """Generic Agent class. To be used as a parent class for different implementations"""
    
    def __init__(self, name: str=None) -> None:
        self._name = name


    def get_move(self, game: game.ConnectGame) -> None:
        """Returns the Agent's next move, based on the game_state
        
        Args:
            game (ConnectGame): the ConnectGame instance this Agent is playing in
        """

        pass


    def handle_invalid_move(self) -> None:
        """Called when the ConnectGame rejects the proposed move"""
        
        pass


class Human(Agent):
    """Agent that prompts for user input each move. No independent decisions.

    Each turn the game board is printed, and the user is prompted to
    enter their next move using the keyboard.
    """

    def __init__(self, name: str=None) -> None:
        self._name = name


    def get_move(self, game: game.ConnectGame) -> np.ndarray:
        """Prompts the user for their next move.

        Prints the current game board to the screen and prompts the user
        for what column to add the piece to. Calculates the row the piece 
        will land in, and submits the move the the ConnectGame. If the move
        is illegal, handle_invalid_move is called.

        Args:
            game (ConnectGame): An instance of the current game.

        Returns:
            An ndarray representing the move, with a 1 in the row,col of the new
            piece, and all other entries zero
        """

        self.print_board(game._game_board, 
                        game._player1._name, 
                        game._player2._name)

        col = input("{}'s move:")

        new_state = self.get_new_state(col, game._game_board)
        return new_state


    def print_board(self, board: np.ndarray, p1: str, p2: str) -> None:
        """Prints the game board to the console.

        Prints the current game board with player one as X and player two as O
        
        Args:
            board (np.ndarray): A 6x7 array for the current board. A 1 represents
                one of player one's pieces. A 2 for player two's
            p1, p2 (str): Names for player one and player two
        """

        for row in board:
            row_str = ['X' if x == 1 else 'O' if x == 2 else '_' for x in row]
            print('|', '|'.join(row_str), '|')

        print('|', '|'.join(range(7)), '|')
    
        print('\n\n{p1}: X, {p2}: O')


    def get_new_state(self, col: int, board: np.ndarray) -> np.ndarray:
        """Calculate the row the piece lands in and return the new game_state

        Args:
            col (int): The column the player chose for their move
            board (np.ndarray): The current board_state

        Returns:
            An ndarray representing the move, with a 1 in the row,col of the new
            piece, and all other entries zero
        """

        
