import numpy as np
import random
import time

COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0


# don't change the class name
class AI(object):
    # chessboard_size, color, time_out passed from agent

    myColor = 0

    def __init__(self, chessboard_size, color, time_out):
        self.chessboard_size = chessboard_size
        # You are white or black
        self.color = color
        AI.myColor = color
        # the max time you should use, your algorithm's run time must not exceed the time limit.
        self.time_out = time_out
        # You need add your decision into your candidate_list. System will get the end of your candidate_list as your
        # decision .
        self.candidate_list = []

    # The input is current chessboard.
    def go(self, chessboard):

        # Clear candidate_list, must do this step
        self.candidate_list.clear()
        # ==================================================================
        # Write your algorithm here
        maxV = -np.Inf
        oldBoard = chessboard.copy()

        self.candidate_list = AI.getMoves(self.chessboard_size, chessboard, self.color)
        length = len(self.candidate_list)
        self.candidate_list = AI.preProcess(self.candidate_list, self.chessboard_size, chessboard, self.color)

        cut = 0
        per = AI.count(self.chessboard_size, chessboard)
        for m in self.candidate_list:
            if cut >= length:
                break
            else:
                cut += 1
            val = 0
            tempX, tempY = m
            AI.place(self.chessboard_size, chessboard, self.color, tempX, tempY)
            # consider different period
            if per <= 18:
                val = -AI.alphaBeta(self.chessboard_size, chessboard, -np.Inf, np.Inf, -self.color, 3)
            elif 18 < per <= 34:
                val = -AI.alphaBeta(self.chessboard_size, chessboard, -np.Inf, np.Inf, -self.color, 2)
            elif 34 < per <= 58:
                val = -AI.alphaBeta(self.chessboard_size, chessboard, -np.Inf, np.Inf, -self.color, 3)
            else:
                val = -AI.alphaBeta(self.chessboard_size, chessboard, -np.Inf, np.Inf, -self.color, 5)

            for i in range(self.chessboard_size):
                for j in range(self.chessboard_size):
                    chessboard[i][j] = oldBoard[i][j]

            if val > maxV:
                maxV = val
                self.candidate_list.append(m)
        return self.candidate_list



    @staticmethod
    def preProcess(candidate_list, size, board, color):
        score = []
        oldBoard = board.copy()
        for m in candidate_list:
            tempX, tempY = m
            AI.place(size, board, color, tempX, tempY)
            temp = AI.evaluate(size, board)
            if m == (0, 0) or m == (0, size-1) or m == (size-1, 0) or m == (size-1, size-1):
                temp += 20000
            score.append(temp)

            for i in range(size):
                for j in range(size):
                    board[i][j] = oldBoard[i][j]

        n = len(score)
        for i in range(n-1):
            k = i
            for j in range(i+1, n):
                if score[k] < score[j] and color == AI.myColor:
                    k = j
                elif score[k] > score[j] and color == -AI.myColor:
                    k = j
            score[i], score[k] = score[k], score[i]
            candidate_list[i], candidate_list[k] = candidate_list[k], candidate_list[i]
        return candidate_list

    @staticmethod
    def count(size, board):
        ans = 0
        for x in range(size):
            for y in range(size):
                if board[x][y] != 0:
                    ans += 1
        return ans

    @staticmethod
    def isValid(size, board, color, x, y):
        if x < 0 or x >= size or y < 0 or y >= size:
            return False
        else:
            DIR = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
            for direction in range(8):
                dx = DIR[direction][0]
                dy = DIR[direction][1]
                tempX = x + dx
                tempY = y + dy
                while 0 <= tempX < size and 0 <= tempY < size and board[tempX][tempY] == -color:
                    tempX += dx
                    tempY += dy
                if 0 <= tempX < size and 0 <= tempY < size and board[tempX][tempY] == color:
                    tempX -= dx
                    tempY -= dy
                    if tempX == x and tempY == y:
                        continue
                    return True
            return False

    @staticmethod
    def getMoves(size, board, color):
        moves = []
        for i in range(size):
            for j in range(size):
                if board[i][j] == 0:
                    if AI.isValid(size, board, color, i, j):
                        moves.append((i, j))
        return moves

    @staticmethod
    def canMove(size, board, color):
        for i in range(size):
            for j in range(size):
                if board[i][j] == 0:
                    if AI.isValid(size, board, color, i, j):
                        return True
        return False

    @staticmethod
    def evaluate(size, board):
        mobility = 0
        outside = 0
        inside = 0
        corner = 0
        xLocation = 0
        stable = 0
        map_val = 0
        number = 0
        opp_mob = my_mob = 0
        opp_out = my_out = 0
        opp_in = my_in = 0
        opp_num = my_num = 0

        WEIGHTS = np.array([
            [220, -20, 115, 85, 85, 115, -20, 220],
            [-20, -80, -40, 10, 10, -40, -80, -20],
            [115, -40, 20, 20, 20, 20, -40, 115],
            [85, 13, 20, -29, -29, 20, 13, 85],
            [85, 13, 20, -29, -29, 20, 13, 85],
            [115, -40, 20, 20, 20, 20, -40, 115],
            [-20, -80, -40, 10, 10, -40, -80, -20],
            [220, -20, 115, 85, 85, 115, -20, 220]
        ])
        DIR = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        # number
        for i in range(size):
            for j in range(size):
                if board[i][j] == AI.myColor:
                    my_num += 1
                elif board[i][j] == -AI.myColor:
                    opp_num += 1
        if my_num > opp_num:
            number = (100 * my_num) / (my_num + opp_num + 1.1)
        else:
            number = -(100 * opp_num) / (my_num + opp_num + 1.1)

        # mobility
        for i in range(size):
            for j in range(size):
                if board[i][j] == 0:
                    if AI.isValid(size, board, -AI.myColor, i, j):
                        opp_mob += 1
                    if AI.isValid(size, board, AI.myColor, i, j):
                        my_mob += 1

        if my_mob > opp_mob:
            mobility = (100 * my_mob) / (my_mob + opp_mob + 1.1)
        else:
            mobility = -(100 * opp_mob) / (my_mob + opp_mob + 1.1)

        # outside
        for x in range(size):
            for y in range(size):
                if board[x][y] != 0:
                    for d in range(8):
                        tempX = x + DIR[d][0]
                        tempY = y + DIR[d][1]
                        if 0 <= tempX < size and 0 <= tempY < size and board[tempX][tempY] == 0:
                            if board[x][y] == AI.myColor:
                                my_out += 1
                            else:
                                opp_out += 1
        if my_out > opp_out:
            outside = -(100 * my_out) / (my_out + opp_out + 1.1)
        else:
            outside = (100 * opp_out) / (my_out + opp_out + 1.1)

        # inside
        for x in range(size):
            for y in range(size):
                if board[x][y] != 0:
                    flag = True
                    for d in range(8):
                        tempX = x + DIR[d][0]
                        tempY = y + DIR[d][1]
                        if 0 <= tempX < size and 0 <= tempY < size and board[tempX][tempY] == 0:
                            flag = False
                            break
                    if flag:
                        if board[x][y] == AI.myColor:
                            my_in += 1
                        else:
                            opp_in += 1
        if my_in > opp_in:
            inside = (100 * my_in) / (my_in + opp_in + 1.1)
        else:
            inside = -(100 * opp_in) / (my_in + opp_in + 1.1)

        # corner
        if board[0][0] != 0:
            if board[0][0] == AI.myColor:
                corner += 1
            else:
                corner -= 1
        if board[size - 1][size - 1] != 0:
            if board[size - 1][size - 1] == AI.myColor:
                corner += 1
            else:
                corner -= 1
        if board[0][size - 1] != 0:
            if board[0][size - 1] == AI.myColor:
                corner += 1
            else:
                corner -= 1
        if board[size - 1][0] != 0:
            if board[size-1][0] == AI.myColor:
                corner += 1
            else:
                corner -= 1

        # x location
        if board[0][0] == 0:
            if board[0][1] == AI.myColor:
                xLocation += 0.67
            elif board[0][1] == -AI.myColor:
                xLocation -= 0.67

            if board[1][0] == AI.myColor:
                xLocation += 0.67
            elif board[1][0] == -AI.myColor:
                xLocation -= 0.67

            if board[1][1] == AI.myColor:
                xLocation += 1
            elif board[1][1] == -AI.myColor:
                xLocation -= 1

        if board[size - 1][size - 1] == 0:
            if board[size - 1][size - 2] == AI.myColor:
                xLocation += 0.67
            elif board[size - 1][size - 2] == -AI.myColor:
                xLocation -= 0.67

            if board[size - 2][size - 1] == AI.myColor:
                xLocation += 0.67
            elif board[size - 2][size - 1] == -AI.myColor:
                xLocation -= 0.67

            if board[size - 2][size - 2] == AI.myColor:
                xLocation += 1
            elif board[size - 2][size - 2] == -AI.myColor:
                xLocation -= 1

        if board[0][size - 1] == 0:
            if board[0][size - 2] == AI.myColor:
                xLocation += 0.67
            elif board[0][size - 2] == -AI.myColor:
                xLocation -= 0.67

            if board[1][size - 1] == AI.myColor:
                xLocation += 0.67
            elif board[1][size - 1] == -AI.myColor:
                xLocation -= 0.67

            if board[1][size - 2] == AI.myColor:
                xLocation += 1
            elif board[1][size - 2] == -AI.myColor:
                xLocation -= 1

        if board[size - 1][0] == 0:
            if board[size - 1][1] == AI.myColor:
                xLocation += 0.67
            elif board[size - 1][1] == -AI.myColor:
                xLocation -= 0.67

            if board[size - 2][0] == AI.myColor:
                xLocation += 0.67
            elif board[size - 2][0] == -AI.myColor:
                xLocation -= 0.67

            if board[size - 2][1] == AI.myColor:
                xLocation += 1
            elif board[size - 2][1] == -AI.myColor:
                xLocation -= 1

        # stable
        for x in range(0, size):  # No blank at all directions
            for y in range(0, size):
                if board[x][y] == 0:
                    continue
                isStable = True
                for d in range(8):
                    tempX = x
                    tempY = y
                    while 0 <= tempX < size and 0 <= tempY < size:
                        if board[tempX][tempY] == 0:
                            isStable = False
                            break
                        tempX += DIR[d][0]
                        tempY += DIR[d][1]
                if isStable:
                    if board[x][y] == AI.myColor:
                        stable += 1
                    else:
                        stable -= 1

        if board[0][0] == AI.myColor:  # corner and the pieces next to it are stable
            i = 1
            while i < size - 1 and board[0][i] == AI.myColor:
                stable += 1
                i += 1
            i = 1
            while i < size - 1 and board[i][0] == AI.myColor:
                stable += 1
                i += 1

        if board[0][size - 1] == AI.myColor:
            i = size - 2
            while i >= 1 and board[0][i] == AI.myColor:
                stable += 1
                i -= 1
            i = 1
            while i < size - 1 and board[i][size - 1] == AI.myColor:
                stable += 1
                i += 1

        if board[size - 1][0] == AI.myColor:
            i = size - 2
            while i >= 1 and board[i][0] == AI.myColor:
                stable += 1
                i -= 1
            i = 1
            while i < size - 1 and board[size - 1][i] == AI.myColor:
                stable += 1
                i += 1

        if board[size - 1][size - 1] == AI.myColor:
            i = size - 2
            while i >= 1 and board[size - 1][i] == AI.myColor:
                stable += 1
                i -= 1
            i = size - 2
            while i >= 1 and board[i][size - 1] == AI.myColor:
                stable += 1
                i -= 1

        if board[0][0] == -AI.myColor:  # opposite
            i = 1
            while i < size - 1 and board[0][i] == -AI.myColor:
                stable -= 1
                i += 1
            i = 1
            while i < size - 1 and board[i][0] == -AI.myColor:
                stable -= 1
                i += 1

        if board[0][size - 1] == -AI.myColor:
            i = size - 2
            while i >= 1 and board[0][i] == -AI.myColor:
                stable -= 1
                i -= 1
            i = 1
            while i < size - 1 and board[i][size - 1] == -AI.myColor:
                stable -= 1
                i += 1

        if board[size - 1][0] == -AI.myColor:
            i = size - 2
            while i >= 1 and board[i][0] == -AI.myColor:
                stable -= 1
                i -= 1
            i = 1
            while i < size - 1 and board[size - 1][i] == -AI.myColor:
                stable -= 1
                i += 1

        if board[size - 1][size - 1] == -AI.myColor:
            i = size - 2
            while i >= 1 and board[size - 1][i] == -AI.myColor:
                stable -= 1
                i -= 1
            i = size - 2
            while i >= 1 and board[i][size - 1] == -AI.myColor:
                stable -= 1
                i -= 1

        # map weight
        for i in range(size):
            for j in range(size):
                if board[i][j] == AI.myColor:
                    map_val += WEIGHTS[i][j]
                elif board[i][j] == -AI.myColor:
                    map_val -= WEIGHTS[i][j]
        value = 0
        state = AI.count(size, board)
        if state > 61:
            value = 200 * number + map_val
        elif 0 < state <= 20:
            value = 10*number + 20890*corner - 7877*xLocation + 89*mobility + 72*stable + 74*outside + map_val
        elif 20 < state <= 49:
            value = 10*number + 21350*corner - 7985*xLocation + 150*mobility + 125*stable + 75*outside + 11*inside + map_val
        else:
            value = 13*number + 20060*corner - 7595*xLocation + 58*mobility + 310*stable + 58*outside + 13*inside + map_val

        return value

    @staticmethod
    def place(size, board, color, x, y):
        if x < 0 or x >= size or y < 0 or y >= size:
            return
        board[x][y] = color
        DIR = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for d in range(8):
            tempX = x + DIR[d][0]
            tempY = y + DIR[d][1]
            while 0 <= tempX < size and 0 <= tempY < size and board[tempX][tempY] == -color:
                tempX += DIR[d][0]
                tempY += DIR[d][1]
            if 0 <= tempX < size and 0 <= tempY < size and board[tempX][tempY] == color:
                while True:
                    tempX -= DIR[d][0]
                    tempY -= DIR[d][1]
                    if (tempX, tempY) == (x, y):
                        break
                    board[tempX][tempY] = color

    @staticmethod
    def alphaBeta(size, board, alpha, beta, color, depth):
        maxV = -np.Inf
        sign = -1
        if color == AI.myColor:
            sign = 1

        if depth <= 0:
            return sign*AI.evaluate(size, board)

        if not AI.canMove(size, board, color):
            if not AI.canMove(size, board, -color):
                return sign*AI.evaluate(size, board)
            return -AI.alphaBeta(size, board, -beta, -alpha, -color, depth)

        moves = AI.getMoves(size, board, color)
        moves = AI.preProcess(moves, size, board, color)
        oldBoard = board.copy()
        for m in moves:
            tempX, tempY = m
            AI.place(size, board, color, tempX, tempY)
            val = -AI.alphaBeta(size, board, -beta, -alpha, -color, depth-1)
            for i in range(size):
                for j in range(size):
                    board[i][j] = oldBoard[i][j]
            if val > alpha:
                if val >= beta:
                    return val
                alpha = val
            if val > maxV:
                maxV = val
        return maxV


# import datetime
# cd = [[0,0,-1,-1,-1,-1,0,0],
# [1,0,-1,-1,-1,-1,0,0],
# [1,1,-1,-1,-1,-1,0,-1],
# [1,-1,1,-1,1,-1,-1,1],
# [1,-1,1,1,-1,-1,1,0],
# [1,-1,-1,1,1,-1,1,0],
# [1,-1,-1,-1,-1,1,1,0],
# [0,-1,-1,-1,-1,-1,1,-1]]
# cb = np.zeros((8, 8), dtype=np.int)
# for i in range(8):
#     for j in range(8):
#         cb[i][j] = cd[i][j]
# # cb[3][4], cb[4][3], cb[3][3], cb[4][4] = COLOR_BLACK, COLOR_BLACK, COLOR_WHITE, COLOR_WHITE
# ai = AI(8, 1, 20)
# print(cb)
# ai.go(cb)
# print(ai.candidate_list)


