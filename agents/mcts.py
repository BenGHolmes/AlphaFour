import numpy as np
from agents import Agent
import time
import math
import helpers

class Node(object):
    def __init__(self, game_board: np.ndarray):
        self.state = game_board
        self.children = None
        self.value = 0
        self.visits = 0

    def add_children(self, children: np.ndarray):
        self.children = [Node(gb) for gb in children]


class MCTS(Agent):
    """Agent that implements Monte Carlo Tree Search to select next move."""

    NUM_SIMULATIONS = 1000
    EXPLORATION_PARAMETER = np.sqrt(2)

    def __init__(self, name: str = None) -> None:
        self._name = name


    def get_move(self, game_board: np.ndarray) -> np.ndarray:
        # Initialize root to the current state and populate children. 
        root = Node(game_board)
        root.add_children(root.state + helpers.get_legal_moves(root.state))

        for i in range(self.NUM_SIMULATIONS):
            path = [root]  # For storing nodes we traverse along the way
            node = root
            while node.children is not None:
                # Select child that maximizes UCT score
                children = node.children
                max_score = -np.inf
                new_move = []
                for child in children:
                    uct_score = self.get_uct_score(child.value, child.visits, node.visits)
                    if uct_score > max_score:
                        max_score = uct_score
                        new_move = [child]
                    elif uct_score == max_score:
                        # Add to list so we can randomly sample from tied states
                        new_move.append(child)

                # Update node and add to path
                node = new_move[np.random.randint(0, len(new_move))]
                path.append(node)

            value = self.get_static_value(node.state)
            
            if value is not None:
                # If value is not None, this is an end-game state. Update visits, values, etc.
                for i,node in enumerate(path):
                    node.value += ((-1)**i) * value
                    node.visits += 1
            else: 
                # If value is None, expand, choose random move, simulate to end
                # Flip sign since player changes each move
                node.add_children(-node.state + helpers.get_legal_moves(node.state))

                first_child = node.children[np.random.randint(0, len(node.children))]
                sim_board = first_child.state
                move_idx = 0
                while self.get_static_value(sim_board) is None:
                    moves = helpers.get_legal_moves(sim_board)
                    sim_board = -sim_board + moves[np.random.randint(0, len(moves))]
                    move_idx += 1

                # Value relative to current player. If move_idx is odd, this is negative
                # of the value of the original simulation node.
                value = self.get_static_value(sim_board)*(-1)**move_idx
                
                path.append(first_child)
                for i,node in enumerate(path):
                    node.value += ((-1)**i) * value
                    node.visits += 1

        # Choose most visited move
        max_visits = 0
        max_value = None
        move = None
        for node in root.children:
            print(f"Visits: {node.visits} | Value: {node.value/node.visits}")
            print(node.state, '\n')

            if node.visits > max_visits:
                move = node.state - game_board
                max_visits = node.visits
                max_value = node.value

        # print(f"Found best move with {max_visits} visits and a value of {max_value}")

        return move


    def get_uct_score(self, w: float, n: int, N: int):
        """Returns the UCT score of a node with a score of w, n visits and N parent visits.
        
        See: https://en.wikipedia.org/wiki/Monte_Carlo_tree_search#Exploration_and_exploitation
        
        Args:
            w (float): Current score of the node being evaluated.
            n (int): Number of visits to current node.
            N (int): Number of visits to parent node.
        
        Returns:
            UCT score as defined above.
        """
        if not n:
            # If n is zero, return inf
            return np.inf

        return w/float(n) + self.EXPLORATION_PARAMETER*np.sqrt(np.log(N)/float(n))

        

    def get_static_value(self, game_board: np.ndarray) -> float:
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
            return 1
        elif window_sums.min() == -4:
            return -1
        elif helpers.get_legal_moves(game_board).size == 0:
            return 0
        
        return None


    def handle_invalid_move(self) -> None:
        # Throw exception during development
        # TODO: Add some nice handler later on
        raise Exception
        

