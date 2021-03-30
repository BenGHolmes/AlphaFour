from agents import Agent, Human, AlphaBeta, MCTS, AlphaFour
from connectgame import ConnectGame
import argparse

agents = {
    'Human': Human,
    'AlphaBeta': AlphaBeta,
    'MCTS': MCTS,
    'AlphaFour': AlphaFour
}

def play_game():
    p1 = AlphaBeta(name="P1")
    p2 = AlphaBeta(name="P2")

    game = ConnectGame(p1, p2)
    game.play_game()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a game of Connect Four between two agents.')
    parser.add_argument('-p1','--player1',nargs='+', default=['Human', 'Human'])
    parser.add_argument('-p2','--player2',nargs='+', default=['AlphaBeta', 'AlphaBeta'])

    args = parser.parse_args()

    p1_type, p1_name = (args.player1 + ['P1'])[:2]
    p2_type, p2_name = (args.player2 + ['P2'])[:2]

    if p1_type not in agents.keys():
        print(f"Unknown Agent: {p1_type}")
        exit(1)
    if p2_type not in agents.keys():
        print(f"Unknown Agent: {p2_type}")
        exit(1)

    p1 = agents[p1_type](name=p1_name)
    p2 = agents[p2_type](name=p2_name)

    game = ConnectGame(p1,p2)
    game.play_game()


    
    