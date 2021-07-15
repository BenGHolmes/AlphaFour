import numpy as np
from agents import Agent
import time
import math
from random import choice
from time import time
from connectboard import ConnectBoard


class Model(object):
    """The actual AlphaFour Neural Network.

    TODO: All the challenging stuff
    """

    def value(self, state):
        # FIXME: Implement the NN
        return 2 * np.random.rand() - 1

    def policy(self, state):
        # FIXME: Implement the NN
        return np.full(7, 1 / 7)


class Node(object):
    """Node object used in MCTS."""

    def __init__(self, state):
        self.state = state  # Current state
        self.P = np.zeros(7)  # Probability of taking each action from this state
        self.W = np.zeros(7)  # Total value of next state from N visits
        self.N = np.zeros(7)  # Number of times each action was taken
        self.children = None  # Children of this Node

    def add_children(self, moves):
        """Add a child for each of the given moves."""
        self.children = [None for _ in range(7)]

        # Boolean for which columns have a legal move
        col_has_move = moves.sum(axis=0).sum(axis=0)

        move_idx = 0
        for i in range(7):
            if col_has_move[i]:
                next_state = AlphaFour.next_game_state(self.state, moves[move_idx])
                self.children[i] = Node(next_state)
                move_idx += 1

    def ucb_score(self, exploration_constant: float) -> np.array:
        """Return the UCB score of each edge from this node."""
        ucb = np.where(
            self.N > 0,
            self.W / self.N + exploration_constant * self.P / (1 + self.N),
            np.inf,
        )

        # Avoid invalid moves by setting UCB to -inf for None children
        ucb = np.where(self.children is not None, ucb, -np.inf)

        return ucb


class AlphaFour(Agent):
    """Agent that implements a lightweight version of the AlphaZero/AlphaGo algorithm.

    TODO:
        - Store subtree so I don't have to rebuild every time
        - General performance boosts. Pretty slow going right now
    """

    def __init__(self) -> None:
        self._EXPLORATION_CONSTANT = 1
        self._NUM_MCTS = 100
        self.model = Model()

    def select(self, node: Node) -> tuple[Node, list[(Node, int)]]:
        """
        """
        path = []  # For storing nodes we traverse along the way

        # Loop until we hit a leaf node
        while node.children is not None:
            # UCB score as defined in AlphaGo Zero paper. Use infinity for unvisited nodes
            # see: "Mastering the game of Go without human knowledge"
            ucb = node.ucb_score(self._EXPLORATION_CONSTANT)
            next_move = np.argmax(ucb)

            path.append((node, next_move))

            # Take best move and add to path
            node = node.children[next_move]

        return node, path

    def expand_and_sim(self, node: Node) -> float:
        # Combined expand and simulate. We add children to this node, and assign
        # the value of the node and prior probabilities of possible actions
        node.add_children(ConnectBoard.get_legal_moves(node.state[0] + node.state[1]))
        node.P = self.model.policy(node.state)  # Add prior probabilities
        node.W = self.model.value(node.state)  # Add predicted value of state
        return node.W

    def back_propagate(self, path: list[(Node, int)], reward: int) -> None:
        # Work backwards through path and propagate reward
        for i, (node, action) in enumerate(path[::-1]):
            node.N[action] += 1
            node.W[action] += reward * (-1) ** (i + 1)

    def get_move(self, game_board: np.ndarray) -> np.ndarray:
        """Returns the best move for AlphaFour to take from the current state.

        Runs a Monte Carlo tree search using the trained neural network to predict
        both prior probabilities and state values. After a fixed number of simulations
        it returns the most visited next state.

        Args:
            game_board (np.ndarray): current board with a 1 for current player, -1 for
                opponent, and 0 for open space

        Returns:
            An ndarray representing the move, with a 1 in the row,col of the new
            piece, and all other entries zero.
        """
        raise NotImplementedError

    def get_move_with_prob(self, game_state: np.ndarray) -> tuple[np.ndarray, np.array]:
        """Returns the best move along with the probabilities of each possible move.

        Runs the same MCTS as above, using the trained neural net to predict prior
        probabilities and state values. After the simulations, this returns the optimal
        move (most visited) and the probabilities for each of the 7 possible next moves.
        If any columns are full, the probability for that column is 0.

        Args:
            TODO:

        Returns:
            TODO:
        """
        timing = []
        root = Node(game_state)

        for i in range(self._NUM_MCTS):
            leaf, path = self.select(root)

            winner = ConnectBoard.get_winner(leaf.state[0] - leaf.state[1])
            if winner is not None:
                # If leaf has static value it has no children. Back prop
                self.back_propagate(path, winner)
            else:
                # Leaf has children. Expand and simulate
                value = self.expand_and_sim(leaf)
                if path:
                    self.back_propagate(path, value)

        visits = root.N

        # Mask out invalid moves
        col_has_move = (
            ConnectBoard.get_legal_moves(root.state[0] + root.state[1])
            .sum(axis=0)
            .sum(axis=0)
            .astype(bool)
        )
        visits[~col_has_move] = 0

        action = visits.argmax()
        new_state = root.children[action].state

        move = (new_state[0] - game_state[0]) + (new_state[1] - game_state[1])

        return move, root.P

    def handle_invalid_move(self) -> None:
        # Throw exception during development
        # TODO: Add some nice handler later on
        raise Exception

    @staticmethod
    def get_game_state(game_board: np.ndarray) -> np.ndarray:
        """Returns the NN friendly version of the gameboard.

        Following the format of the AlphaGo Zero paper, the game state has the following format:
            game_state[0]: 6x7 array with a 1 in all positions occupied by this agent
            game_state[1]: 6x7 array with a 1 in all positions occupied by the opponent

        Args:
            game_board: Numpy array of the current board in standard format. With a 1 for this 
            player, -1 for the opponent and a 0 for open spaces.

        Returns:
            Numpy array of the NN friendly game state as defined above.
        """
        return np.array(
            [np.where(game_board == 1, 1, 0), np.where(game_board == -1, 1, 0)]
        )

    @staticmethod
    def next_game_state(game_state: np.ndarray, move: np.ndarray) -> np.ndarray:
        """Returns the new game state after the given move is made by the current player.
        
        Args:
            game_state: Numpy array of the current game state.
            move: Numpy array of the move to be made by the current player. 
        
        Returns:
            Numpy array of the new game state after the move is made and the current
            player is changed.
        """
        curr_player_pieces = game_state[0] + move
        opponent_pieces = game_state[1]

        return np.array([opponent_pieces, curr_player_pieces])
