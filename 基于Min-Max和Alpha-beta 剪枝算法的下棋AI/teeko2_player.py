import random
import numpy as np
from numpy import *
import copy

class Teeko2Player:
    """ An object representation for an AI game player for the game Teeko2.
    """
    board = [[' ' for j in range(5)] for i in range(5)]
    pieces = ['b', 'r']

    def __init__(self):
        """ Initializes a Teeko2Player object by randomly selecting red or black as its
        piece color.
        """
        self.my_piece = random.choice(self.pieces)
        self.opp = self.pieces[0] if self.my_piece == self.pieces[1] else self.pieces[1]

    # 找到当前state可能的下一个state
    def succ(self, state, piece):
        # determine phase
        drop_phase = True
        numai = self.count(state, self.my_piece)
        numopp = self.count(state, self.opp)
        if numai >= 4 and numopp >= 4:
            drop_phase = False

        # during drop phase
        if drop_phase:
            res, cur_piece = [], piece
            for i in range(len(state)):
                for j in range(len(state[0])):
                    if state[i][j] == ' ':
                        temp = copy.deepcopy(state)
                        temp[i][j] = piece
                        res.append(temp)
            return res

        # during move phase
        else:
            res, cur_piece = [], piece
            for i in range(len(state)):
                for j in range(len(state[0])):
                    possible_move = [[i + 1, j], [i, j + 1], [i - 1, j], [i, j - 1], [i - 1, j - 1], [i - 1, j + 1],
                                     [i + 1, j + 1], [i + 1, j - 1]]
                    if state[i][j] == cur_piece:
                        for moves in possible_move:
                            row, col = moves[0], moves[1]
                            if 0 < row < len(state) and 0 < col < len(state[0]) and state[row][col] == ' ':
                                temp = copy.deepcopy(state)
                                temp[row][col] = piece
                                temp[i][j] = ' '
                                res.append(temp)
            return res

    # 计算当前state某种棋子的数量
    def count(self, state, piece):
        count = 0
        for row in state:
            for col in row:
                if col == piece:
                    count += 1
        return count

    def make_move(self, state):
        """ Selects a (row, col) space for the next move. You may assume that whenever
        this function is called, it is this player's turn to move.

        Args:
            state (list of lists): should be the current state of the game as saved in
                this Teeko2Player object. Note that this is NOT assumed to be a copy of
                the game state and should NOT be modified within this method (use
                place_piece() instead). Any modifications (e.g. to generate successors)
                should be done on a deep copy of the state.

                In the "drop phase", the state will contain less than 8 elements which
                are not ' ' (a single space character).

        Return:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.

        """

        drop_phase = True
        numai = self.count(state, self.my_piece)
        numopp = self.count(state, self.opp)
        if numopp >= 4 and numai >= 4:
            drop_phase = False

        if not drop_phase:
            # accordingly, the AI will not follow the rules after the drop phase!

            move = []
            value, cur_state = self.Max_Value(state, 0)
            # 生成move，因为返回值是一系列state，需要和当前state对比生成move[(row,col),(source_row,source_col)]
            row, col, source_row, source_col, flag1, flag2 = 0, 0, 0, 0, 0, True
            for i in range(len(state)):
                if not flag2:
                    break
                for j in range(len(state[0])):
                    if state[i][j] != cur_state[i][j] and state[i][j] == ' ':
                        row, col = i, j
                        flag1 += 1
                    if state[i][j] != cur_state[i][j] and cur_state[i][j] == ' ':
                        source_row, source_col = i, j
                        flag1 += 1
                    if flag1 == 2:
                        flag2 = False
                        break

            move.insert(0, (row, col))
            move.insert(1, (source_row, source_col))
            return move

        move = []
        value, cur_state = self.Max_Value(state, 0)
        row, col, flag = 0, 0, True
        for i in range(len(state)):
            if not flag:
                break
            for j in range(len(state[0])):
                if state[i][j] != cur_state[i][j]:
                    row, col = i, j
                    flag = False
                    break
        move.insert(0, (row, col))
        return move

    def Max_Value(self, state, depth):
        temp = state
        # reach terminate point
        if self.game_value(state) != 0:
            return (self.game_value(state), state)
        # reach max depth but game continue
        # 目前把深度控制在3，这样每一步的时间可以控制在4秒以内
        elif depth >= 3:
            return self.heuristic_game_value(state, self.my_piece)

        else:
            alpha = float('-Inf')
            for moves in self.succ(state, self.my_piece):
                score = self.Min_Value(moves, depth + 1)
                # 剪枝，只有比目前分数高的子节点才有资格继续扩展
                if score[0] > alpha:
                    alpha = score[0]
                    temp = moves
        return alpha, temp

    def Min_Value(self, state, depth):
        temp = state
        if self.game_value(state) != 0:
            return (self.game_value(state), state)
        elif depth >= 3:
            return self.heuristic_game_value(state, self.opp)
        else:
            beta = float('Inf')
            for moves in self.succ(state, self.opp):
                score = self.Max_Value(moves, depth + 1)
                if score[0] < beta:
                    beta = score[0]
                    temp = moves
        return beta, temp

    def opponent_move(self, move):
        """
        Args:
            move (list): a list of move tuples such that its format is
                    [(row, col), (source_row, source_col)]
                where the (row, col) tuple is the location to place a piece and the
                optional (source_row, source_col) tuple contains the location of the
                piece the AI plans to relocate (for moves after the drop phase). In
                the drop phase, this list should contain ONLY THE FIRST tuple.
        """
        # validate input
        if len(move) > 1:
            source_row = move[1][0]
            source_col = move[1][1]
            if source_row != None and self.board[source_row][source_col] != self.opp:
                raise Exception("You don't have a piece there!")
        if self.board[move[0][0]][move[0][1]] != ' ':
            raise Exception("Illegal move detected")
        if len(move) > 1 and pow(move[0][0] - move[1][0], 2) + pow(move[0][1] - move[1][1], 2) > 2:
            raise Exception("Illegal move detected")
        # make move
        self.place_piece(move, self.opp)

    def place_piece(self, move, piece):
        if len(move) > 1:
            self.board[move[1][0]][move[1][1]] = ' '
        self.board[move[0][0]][move[0][1]] = piece

    def print_board(self):
        """ Formatted printing for the board """
        for row in range(len(self.board)):
            line = str(row) + ": "
            for cell in self.board[row]:
                line += cell + " "
            print(line)
        print("   A B C D E")


    # def heuristic_game_value(self, state, piece):
    #     # more pieces on the same row, col, diagonal or a diamond --> higher score for the piece
    #     if piece == 'b':
    #         my_piece = 'b'
    #         opp = 'r'
    #
    #     else:
    #         my_piece = 'r'
    #         opp = 'b'
    #
    #     my_pos,opp_pos=[],[]
    #
    #     # record position of each kind of pieces on current state
    #     for row in range(len(state)):
    #         for col in range(len(state[0])):
    #             if state[row][col]==my_piece:
    #                 my_pos.append([row,col])
    #             elif state[row][col]==opp:
    #                 opp_pos.append([row, col])
    #
    #
    #     maxai, maxopp = 0, 0
    #
    #     # move to same row
    #     # count the min distance from the piece to the line
    #     for row in range(len(state)):
    #         my_count, opp_count = 0, 0
    #         # my_piece
    #         for marker in my_pos:
    #             my_count+=abs(row-marker[0])
    #         for marker in opp_pos:
    #             opp_count+=abs(row-marker[0])
    #         # more required moves leads to lower score
    #         if my_count>opp_count:
    #             maxopp+=1
    #         elif my_count<opp_count:
    #             maxai+=1
    #
    #     # move to same col
    #     # count the min distance from the piece to the line
    #     for col in range(len(state[0])):
    #         my_count, opp_count = 0, 0
    #         for marker in my_pos:
    #             my_count+=abs(col-marker[1])
    #         for marker in opp_pos:
    #             opp_count+=abs(col-marker[1])
    #         # more required moves leads to lower score
    #         if my_count>opp_count:
    #             maxopp+=1
    #         elif my_count<opp_count:
    #             maxai+=1
    #
    #
    #
    #     # on diagnal \
    #     # count the min distance from the piece to the line
    #     for row in range(2):
    #         for col in range(2):
    #             my_count, opp_count = 0, 0
    #             for marker in my_pos:
    #                 my_count+=(abs(col-marker[1])*abs(row-marker[0]))/self.distance([row,col],[row+3,col+3])
    #             for marker in opp_pos:
    #                 opp_count += (abs(col - marker[1]) * abs(row - marker[0])) / self.distance([row, col],[row + 3, col + 3])
    #             # more required moves leads to lower score
    #             if my_count > opp_count:
    #                 maxopp += 1
    #             elif my_count < opp_count:
    #                 maxai += 1
    #
    #
    #     # on diagnal /
    #     for row in range(2):
    #         for col in range(3,5):
    #             my_count, opp_count = 0, 0
    #             for marker in my_pos:
    #                 my_count+=(abs(col-marker[1])*abs(row-marker[0]))/self.distance([row,col],[row-3,col-3])
    #             for marker in opp_pos:
    #                 opp_count += (abs(col - marker[1]) * abs(row - marker[0])) / self.distance([row, col],[row - 3, col - 3])
    #             # more required moves leads to lower score
    #             if my_count > opp_count:
    #                 maxopp += 1
    #             elif my_count < opp_count:
    #                 maxai += 1
    #
    #
    #     # move to diamond
    #     # count the min distance from the piece to diamond
    #     for row in range(1, len(state) - 1):
    #         for col in range(1, len(state[0]) - 1):
    #             my_count, opp_count = 0, 0
    #             win_position=[[row+1,col],[row-1,col],[row,col+1],[row,col-1]]
    #             for marker in my_pos:
    #                 min_dis=99
    #                 for i in range(4):
    #                     distance=self.distance(marker,win_position[i])
    #                     min_dis=min(min_dis,distance)
    #                 my_count+=min_dis
    #             for marker in opp_pos:
    #                 min_dis = 99
    #                 for i in range(4):
    #                     distance = self.distance(marker, win_position[i])
    #                     min_dis = min(min_dis, distance)
    #                 my_count += min_dis
    #             # more required moves leads to lower score
    #             if my_count > opp_count:
    #                 maxopp += 1
    #             elif my_count < opp_count:
    #                 maxai += 1
    #     # the possible maximum score is 27
    #     score=[0,maxopp,maxai,27]
    #
    #     if maxai == maxopp:
    #         return 0, state
    #     elif maxai > maxopp:
    #         return self.normalization(maxai,max(score),min(score),mean(score)), state
    #     else:
    #         return (-1) * self.normalization(maxopp,max(score),min(score),mean(score)), state

    def heuristic_game_value(self, state, piece):
        # 尝试将棋子之间的紧密程度作为评分标准
        # score=20-某一个棋子到其他同种棋子的距离
        # 取最小值
        if piece == 'b':
            my_piece = 'b'
            opp = 'r'

        else:
            my_piece = 'r'
            opp = 'b'


        my_p,opp_p=[],[]
        for row in range(len(state)):
            for col in range(len(state[0])):
                # 记录某种棋子坐标
                if state[row][col]==my_piece:
                    my_p.append((row,col))
                elif state[row][col]==opp:
                    opp_p.append((row,col))

        my_score, opp_score = 20, 20
        maxai, maxopp = 0, 0
        for i in my_p:
            for j in my_p:
                if i != j:
                    my_score -= self.distance(i, j)
            maxai = max(my_score, maxai)

        for i in opp_p:
            for j in opp_p:
                if i != j:
                    opp_score -= self.distance(i, j)
            maxopp = max(opp_score, maxopp)

        # the possible maximum distance is 15.65685424949238
        # the possible minimum distance is 3.414213562373095
        score=[20-15.65685424949238,my_score,opp_score,20-3.414213562373095]

        if my_score == opp_score:
            return 0, state
        elif my_score > opp_score:
            return self.normalization(my_score,max(score),min(score),mean(score)), state
        else:
            return (-1) * self.normalization(opp_score,max(score),min(score),mean(score)), state


    def distance(self,p1, p2):
        return math.sqrt(math.pow((p2[0] - p1[0]), 2) + math.pow((p2[1] - p1[1]), 2))

    def normalization(self,data,maxm,mini,mean):
        return (data-mean)/(maxm-mini)

    def game_value(self, state):
        """ Checks the current board status for a win condition

        Args:
        state (list of lists): either the current state of the game as saved in
            this Teeko2Player object, or a generated successor state.

        Returns:
            int: 1 if this Teeko2Player wins, -1 if the opponent wins, 0 if no winner

        """
        # check horizontal wins
        for row in state:
            for i in range(2):
                if row[i] != ' ' and row[i] == row[i + 1] == row[i + 2] == row[i + 3]:
                    print('1111')
                    return 1 if row[i] == self.my_piece else -1

        # check vertical wins
        for col in range(5):
            for i in range(2):
                if state[i][col] != ' ' and state[i][col] == state[i + 1][col] == state[i + 2][col] == state[i + 3][
                    col]:
                    print('2222')
                    return 1 if state[i][col] == self.my_piece else -1

        # check \ diagonal wins
        for row in range(2):
            for col in range(2):
                if state[row][col] != ' ' and state[row][col] == state[row + 1][col + 1] == state[row + 2][col + 2] == \
                        state[row + 3][col + 3]:
                    print('3333')
                    return 1 if state[row][col] == self.my_piece else -1

        # check / diagonal wins
        for row in range(2):
            for col in range(3,5):
                if state[row][col] != ' ' and state[row][col] == state[row + 1][col - 1] == state[row + 2][col - 2] == \
                        state[row + 3][col - 3]:
                    print((row,col))
                    print('4444')
                    return 1 if state[row][col] == self.my_piece else -1

        # check diamond wins
        for i in range(1, len(state) - 1):
            for j in range(1, len(state[0]) - 1):
                if state[i][j] == ' ' and state[i + 1][j] != ' ' and state[i + 1][j] == state[i - 1][j] == state[i][
                    j + 1] == state[i][j - 1]:
                    print('5555')
                    return 1 if state[i + 1][j] == self.my_piece else -1

        return 0  # no winner yet


# ############################################################################
# #
# # THE FOLLOWING CODE IS FOR SAMPLE GAMEPLAY ONLY
# #
# ############################################################################
#
def main():
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
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                player_move = input("Move (e.g. B3): ")
                while player_move[0] not in "ABCDE" or player_move[1] not in "01234":
                    player_move = input("Move (e.g. B3): ")
                try:
                    ai.opponent_move([(int(player_move[1]), ord(player_move[0])-ord("A"))])
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
            print(ai.my_piece+" moved from "+chr(move[1][1]+ord("A"))+str(move[1][0]))
            print("  to "+chr(move[0][1]+ord("A"))+str(move[0][0]))
        else:
            move_made = False
            ai.print_board()
            print(ai.opp+"'s turn")
            while not move_made:
                move_from = input("Move from (e.g. B3): ")
                while move_from[0] not in "ABCDE" or move_from[1] not in "01234":
                    move_from = input("Move from (e.g. B3): ")
                move_to = input("Move to (e.g. B3): ")
                while move_to[0] not in "ABCDE" or move_to[1] not in "01234":
                    move_to = input("Move to (e.g. B3): ")
                try:
                    ai.opponent_move([(int(move_to[1]), ord(move_to[0])-ord("A")),
                                    (int(move_from[1]), ord(move_from[0])-ord("A"))])
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


if __name__ == "__main__":
    main()
