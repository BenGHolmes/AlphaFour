# AlphaFour
AlphaFour is a Connect Four AI built with Python and TensorFlow, that selects moves using a nerual network and a Monte Carlo search tree similar to DeepMind's AlphaGo and AlphaZero. Connect is a solved game, so this might be too easy for a computer, but it's a way to get my feet wet before trying chess. 

# Getting Started (For Now)
To play a game, run:
~~~
play_game.py
~~~

Both players are currently humans, so this isn't very exciting.

# Getting Started (The Goal)
To play a game, simply run:
 ~~~
 play_game.py [-p1 <agent1> <name>] [-p2 <agent2> <name>]
 ~~~

`-p1` and `-p2` allow you to specify the `Agent` used for player one and/or player two respectively, with player one making the first move. The following player types are available:
- `AlphaFour`: The AI similar to AlphaGo and AlphaZero. Uses Monte Carlo tree search and a neural network to select moves.
- `AlphaBeta`: Basic algorithm that uses alpha-beta pruning to choose the next move.
- `Human`: Rather than automatically suggesting moves, the agent will prompt the user for the next move. Allows you to play against the AI, or with a friend if you really want to play connect4 and can't be bothered to go out and buy a board.

By default, `play_game.py` will start a game with a `Human` as player one, and `AlphaFour` as player two. You can specify either player to override the default behaviour. The name arguments are optional, but let you have some fun compared to the defualt "player1" and "player2"

## Examples:
Play against the `AlphaBeta` AI, with boring names:
~~~
python play_game.py -p2 AlphaBeta
~~~

Default players, with more accurate names:
~~~
python play_game.py -p1 TinyApeBrain -p2 ComputerOverlord
~~~

Give AlphaFour the first move, and trash talk a little:
~~~
python play_game.py -p1 AlphaFour GlorifiedToaster -p2 Human HasThumbsIsCool
~~~