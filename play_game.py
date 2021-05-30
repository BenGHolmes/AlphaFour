from agents import Agent, Human, AlphaBeta, Mcts, AlphaFour
from connectboard import ConnectBoard
import argparse

agents = {
    'Human': Human,
    'AlphaBeta': AlphaBeta,
    'Mcts': Mcts,
    'AlphaFour': AlphaFour
}


def play(p1: Agent, p2: Agent) -> None:
    board = ConnectBoard()
    turn = 0

    while board.winner() is None:
        print(board)

        if turn % 2 == 0:
            p1_board_state = board.current_state()
            move = p1.get_move(p1_board_state)
        else:
            # Invert state so P2 is 1 and P1 is -1
            p2_board_state = -board.current_state()

            # Invert move so we place a -1 on the game board
            move = -p2.get_move(p2_board_state)

        try:
            board.make_move(move)
            turn += 1
        except:
            if turn % 2 == 0:
                p1.handle_invalid_move()
            else:
                p2.handle_invalid_move()

    print(board)
    
    winner = board.winner()
    if winner:
        print("P{} wins!".format(winner))
    else:
        print("Tie!")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a game of Connect Four between two agents.')
    parser.add_argument('-p1','--player1', default='Human')
    parser.add_argument('-p2','--player2', default='AlphaBeta')

    args = parser.parse_args()

    print(args.player1, args.player2)

    p1_type = args.player1
    p2_type = args.player2

    if p1_type not in agents.keys():
        print(f"Unknown Agent: {p1_type}")
        exit(1)
    if p2_type not in agents.keys():
        print(f"Unknown Agent: {p2_type}")
        exit(1)

    p1 = agents[p1_type]()
    p2 = agents[p2_type]()

    play(p1, p2)

    