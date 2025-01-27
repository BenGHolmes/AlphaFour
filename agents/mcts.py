import numpy as np
from agents import Agent
from connectboard import ConnectBoard
import time
import math
from random import choice


class Node(object):
    def __init__(self, game_board: np.ndarray):
        self.state = game_board
        self.children = None
        self.value = 0
        self.visits = 0

    def add_children(self, children: np.ndarray):
        self.children = [Node(gb) for gb in children]


class Mcts(Agent):
    """Agent that implements Monte Carlo Tree Search to select next move."""

    NUM_SIMULATIONS = 2000
    EXPLORATION_PARAMETER = np.sqrt(2)

    def select(self, node):
        path = [node]  # For storing nodes we traverse along the way

        # Loop until we hit a leaf node
        while node.children is not None:
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

        return node, path

    def expand(self, node):
        # Create children. Flip state after move since convention is for current player to be 1
        # and opponent to be -1.
        node.add_children(-node.state + ConnectBoard.get_legal_moves(node.state))
        return choice(node.children)

    def simulate(self, node):
        board = node.state
        turn = 0
        while self.get_static_value(board) is None:
            moves = ConnectBoard.get_legal_moves(board)
            board = -board + choice(moves)
            turn += 1

        return self.get_static_value(board) * (-1) ** turn

    def back_propagate(self, path, reward):
        # Work backwards through path and propagate reward
        for i, node in enumerate(path[::-1]):
            node.visits += 1
            node.value += reward * (-1) ** (i)

    def get_move(self, game_board):
        # Initialize root to the current state and populate children.
        root = Node(game_board)

        for i in range(self.NUM_SIMULATIONS):
            leaf, path = self.select(root)
            if self.get_static_value(leaf.state) is not None:
                # If leaf has static value it has no children. Back prop
                self.back_propagate(path, self.get_static_value(leaf.state))
            else:
                # Leaf has children. Expand and simulate
                new_leaf = self.expand(leaf)
                path.append(new_leaf)

                value = self.simulate(new_leaf)

                self.back_propagate(path, value)

        # Choose most visited move
        max_visits = 0
        max_value = None
        move = None

        for node in root.children:
            if node.visits > max_visits:
                move = abs(node.state + game_board)
                max_visits = node.visits
                max_value = node.value

        print(f"Found best move with {max_visits} visits and a value of {max_value}")
        print(move)

        return move

    def get_uct_score(self, w, n, N):
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

        return w / float(n) + self.EXPLORATION_PARAMETER * np.sqrt(np.log(N) / float(n))

    def get_static_value(self, game_board):
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
        if (game_board == 0).all():
            return None

        windows = game_board.flatten()[ConnectBoard.WINDOW_INDICES].reshape(-1, 4)
        uncontested_windows = windows[windows.min(axis=1) != -windows.max(axis=1)]
        if uncontested_windows.size == 0:
            return 0

        window_sums = uncontested_windows.sum(axis=1)

        if window_sums.max() == 4:
            return 1
        elif window_sums.min() == -4:
            return -1
        elif ConnectBoard.get_legal_moves(game_board).size == 0:
            return 0

        return None

    def handle_invalid_move(self):
        # Throw exception during development
        # TODO: Add some nice handler later on
        raise Exception
