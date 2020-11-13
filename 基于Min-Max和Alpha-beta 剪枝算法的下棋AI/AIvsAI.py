from teeko2_player import Teeko2Player
from random import choice
import numpy as np



print('Hello, this is Samaritan')
ai = Teeko2Player()
piece_count = 0
turn = 0

# drop phase
while piece_count < 8 and ai.game_value((ai.board))==0:
    # get the player or AI's move
    if ai.my_piece == ai.pieces[turn]:
        ai.print_board()
        move = ai.make_move(ai.board)
        ai.place_piece(move, ai.my_piece)
        print(ai.my_piece+" moved at "+chr(move[0][1]+ord("A"))+str(move[0][0]))
    else:
        ai.print_board()
        move = ai.make_move1(ai.board)
        ai.place_piece(move, ai.opp)
        print(ai.opp+" moved at "+chr(move[0][1]+ord("A"))+str(move[0][0]))

    # update the game variables
    piece_count += 1
    turn += 1
    turn %= 2


# move phase - can't have a winner until all 8 pieces are on the board
while ai.game_value(ai.board) == 0:

    # get the player or AI's move
    if ai.my_piece == ai.pieces[turn]:

        move = ai.make_move(ai.board)
        ai.place_piece(move, ai.my_piece)
        print(ai.my_piece+" moved from "+chr(move[1][1]+ord("A"))+str(move[1][0]))
        print("  to "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        ai.print_board()
    else:

        move = ai.make_move1(ai.board)
        ai.place_piece(move, ai.opp)
        print(ai.opp+" moved from "+chr(move[1][1]+ord("A"))+str(move[1][0]))
        print("  to "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        ai.print_board()

    # update the game variables
    turn += 1
    turn %= 2

ai.print_board()
if ai.game_value(ai.board) == 1:
    print("AI_new wins! Game over.")
else:
    print("AI_old win! Game over.")



