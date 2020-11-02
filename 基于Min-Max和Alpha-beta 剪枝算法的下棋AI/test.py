from teeko2_player import Teeko2Player
from random import choice
import numpy as np



print('Hello, this is Samaritan')
ai = Teeko2Player()
piece_count = 0
turn = 0

row=['A','B','C','D','E']
col=['0','1','2','3','4']
# drop phase
while piece_count < 8 and ai.game_value(ai.board) == 0:

    # get the player or AI's move
    if ai.my_piece == ai.pieces[turn]:
        ai.print_board()
        move = ai.make_move(ai.board)
        ai.place_piece(move, ai.my_piece)
        print(ai.my_piece + " moved at " + chr(move[0][1] + ord("A")) + str(move[0][0]))
    else:
        move_made = False
        ai.print_board()
        print(ai.opp + "'s turn")
        while not move_made:
            flag,pai,lie=False,0,0
            while not flag:
                pai=choice([0,1,2,3,4])
                lie=choice([0,1,2,3,4])
                if ai.board[pai][lie]==' ':
                    flag=True
            player_move = row[pai] + col[lie]

            while player_move[0] not in "ABCDE" or player_move[1] not in "01234":
                player_move = input("Move (e.g. B3): ")
            try:
                ai.opponent_move([(int(player_move[1]), ord(player_move[0]) - ord("A"))])
                move_made = True
            except Exception as e:
                print(e)

    # update the game variables
    piece_count += 1
    turn += 1
    turn %= 2



# move phase - can't have a winner until all 8 pieces are on the board
while ai.game_value(ai.board) == 0:

    # get the player or AI's move
    if ai.my_piece == ai.pieces[turn]:
        ai.print_board()
        move = ai.make_move(ai.board)
        ai.place_piece(move, ai.my_piece)
        print(ai.my_piece + " moved from " + chr(move[1][1] + ord("A")) + str(move[1][0]))
        print("  to " + chr(move[0][1] + ord("A")) + str(move[0][0]))
    else:
        move_made = False
        ai.print_board()
        print(ai.opp + "'s turn")
        while not move_made:
            flag,pai,lie=False,0,0
            while not flag:
                pai=choice([0,1,2,3,4])
                lie=choice([0,1,2,3,4])
                if ai.board[lie][pai]==ai.opp:
                    flag=True
            move_from = row[pai] + col[lie]
            print('move from ',move_from)

            while move_from[0] not in "ABCDE" or move_from[1] not in "01234":
                move_from = input("Move from (e.g. B3): ")

            flag,m=False,[]
            while not flag:
                pos=[[pai+1,lie],[pai-1,lie],[pai,lie+1],[pai,lie-1],[pai+1,lie+1],[pai-1,lie-1],[pai-1,lie+1],[pai+1,lie-1]]
                m=choice(pos)
                if 0<=m[0]<5 and 0<=m[1]<5 and ai.board[m[1]][m[0]]==' ':
                    flag=True
            move_to = row[m[0]] + col[m[1]]
            print('move to ',move_to)

            while move_to[0] not in "ABCDE" or move_to[1] not in "01234":
                move_to = input("Move to (e.g. B3): ")
            try:
                ai.opponent_move([(int(move_to[1]), ord(move_to[0]) - ord("A")),
                                  (int(move_from[1]), ord(move_from[0]) - ord("A"))])
                move_made = True
            except Exception as e:
                print(e)

    # update the game variables
    turn += 1
    turn %= 2

ai.print_board()
if ai.game_value(ai.board) == 1:
    print("AI wins! Game over.")
else:
    print("You win! Game over.")

# ai=Teeko2Player()
# state=[[' ',' ',' ',' ','b'],
#        ['r',' ',' ',' ',' '],
#        [' ','b',' ','r',' '],
#        [' ',' ','b',' ','r'],
#        [' ','r','r','b',' '],]
# #
# # value, cur_state = ai.Max_Value(state, 0)
# # cur_state=np.array(cur_state)
# # print(cur_state)
#
# print(ai.game_value(state))
# print(ai.distance((0,0),(0,1))+ai.distance((0,0),(1,0))+ai.distance((0,0),(1,1)))