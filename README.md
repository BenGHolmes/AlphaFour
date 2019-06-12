# AlphaFour
AlphaFour is a Connect Four AI built with Python and TensorFlow, that selects moves using a nerual network and a Monte Carlo search tree similar to DeepMind's AlphaGo and AlphaZero.

# Getting Started
To play a game, simply run `play_game.py [agent1], [agent2]` where `agent*` is the style of player to use. By default, the following player types are available:
- `AlphaFour`: The AI similar to AlphaGo and AlphaZero. Uses Monte Carlo tree search and a neural network to select moves.
- `AlphaBeta`: Basic algorithm that uses alpha-beta pruning to choose the next move.
- `Human`: Rather than automatically suggesting moves, the agent will prompt the user for the next move. Allows you to play against the AI, or with a friend if you really want to play connect4 and can't be bothered to go out and buy a board.

For example, to play against the `AlphaFour` AI you would run:
~~~
python play_game.py AlphaFour Human
~~~
