from alphafour import Agent, Human, ConnectGame

def play_game():
    p1 = Human(name="Ben")
    p2 = Human(name="P2")

    game = ConnectGame(p1, p2)
    game.play_game()


if __name__ == "__main__":
    play_game()