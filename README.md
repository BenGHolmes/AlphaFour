# AlphaFour
AlphaFour is a Connect Four AI built with Python and Pytorch, that selects moves using a nerual network and a Monte Carlo search tree similar to DeepMind's AlphaGo and AlphaZero. Connect is a solved game, so this might be too easy for a computer, but it's a way to get my feet wet before trying chess. 

# Getting Started
To play a game, simply run:
 ~~~
 python play_game.py [-p1 <agent1> <name>] [-p2 <agent2> <name>]
 ~~~

`-p1` and `-p2` allow you to specify the `Agent` used for player one and/or player two respectively, with player one making the first move. The following player types are available:
- `AlphaBeta`: Basic algorithm that uses alpha-beta pruning to choose the next move.
- `Human`: Rather than automatically suggesting moves, the agent will prompt the user for the next move. Allows you to play against the AI, or with a friend if you really want to play connect4 and can't be bothered to go out and buy a board.

By default, `play_game.py` will start a game with a `Human` as player one, and `AlphaBeta` as player two. You can specify either player to override the default behaviour, and add names for a bit of clarity. Examples:
~~~ 
python play_game.py                                 // Default behaviour. Human vs. AlphaBeta
python play_game.py -p1 AlphaBeta                   // AlphaBeta vs. AlphaBeta
python play_game.py -p1 AlphaBeta Skynet -p2 Human  // Give robot first move
~~~