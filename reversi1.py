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
        totalTime = time_out
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
            if per <= 7:
                val = -AI.alphaBeta(self.chessboard_size, chessboard, -np.Inf, np.Inf, -self.color, 4)
            elif 7 < per <= 47:
                val = -AI.alphaBeta(self.chessboard_size, chessboard, -np.Inf, np.Inf, -self.color, 3)
            elif 47 < per <= 56:
                val = -AI.alphaBeta(self.chessboard_size, chessboard, -np.Inf, np.Inf, -self.color, 4)
            elif 56 < per <= 60:
                val = -AI.alphaBeta(self.chessboard_size, chessboard, -np.Inf, np.Inf, -self.color, 5)
            else:
                val = -AI.alphaBeta(self.chessboard_size, chessboard, -np.Inf, np.Inf, -self.color, 6)

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
            score.append(AI.evaluate(size, board))

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
        actionOfEnemy = 0
        actionOfMine = 0
        stable = 0
        outSide = 0
        inSide = 0
        val = 0
        edge = 0

        WEIGHTS = np.array([
            [200, -30, 110, 80, 80, 110, -30, 200],
            [-30, -70, -40, 10, 10, -40, -70, -30],
            [110, -40, 20, 20, 20, 20, -40, 110],
            [80, 10, 20, -30, -30, 20, 10, 80],
            [80, 10, 20, -30, -30, 20, 10, 80],
            [110, -40, 20, 20, 20, 20, -40, 110],
            [-30, -70, -40, 10, 10, -40, -70, -30],
            [200, -30, 110, 80, 80, 110, -30, 200]
        ])
        DIR = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

        # edge

        for i in range(3, size-2):
            if board[i][0] == AI.myColor:
                edge += 1
            elif board[i][0] == -AI.myColor:
                edge -= 1

            if board[i][size-1] == AI.myColor:
                edge += 1
            elif board[i][size-1] == -AI.myColor:
                edge -= 1

            if board[size-1][i] == AI.myColor:
                edge += 1
            elif board[size-1][i] == -AI.myColor:
                edge -= 1

            if board[0][i] == AI.myColor:
                edge += 1
            elif board[0][i] == -AI.myColor:
                edge -= 1


        # outside:
        for x in range(size):
            for y in range(size):
                flag = False
                if board[x][y] != 0:
                    for d in range(8):
                        tempX = x + DIR[d][0]
                        tempY = y + DIR[d][1]
                        if 0 <= tempX < size and 0 <= tempY < size and board[tempX][tempY] == 0:
                            flag = True
                            break
                    if flag:
                        if board[x][y] == AI.myColor:
                            outSide += 1
                        else:
                            outSide -= 1

        # inside:
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
                            inSide += 1
                        else:
                            inSide -= 1


        # stable :
        if board[0][0] == AI.myColor:      # corner and the pieces next to it are stable
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

        if board[size-1][0] == AI.myColor:
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

        if board[0][0] == -AI.myColor:      # opposite
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

        if board[size-1][0] == -AI.myColor:
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

        # mobility:
        for i in range(size):
            for j in range(size):
                if board[i][j] == 0:
                    if AI.isValid(size, board, -AI.myColor, i, j):
                        actionOfEnemy = actionOfEnemy + 1

                    if AI.isValid(size, board, AI.myColor, i, j):
                        actionOfMine = actionOfMine + 1

        mobility = 0
        if actionOfMine > actionOfEnemy:
            mobility = (7890 * actionOfMine) / (actionOfEnemy + actionOfMine + 0.01)
        else:
            mobility = -(7890 * actionOfEnemy) / (actionOfEnemy + actionOfMine + 0.01)


        # Consider corner and "X" location:
        corner = 0
        if board[0][0] == AI.myColor:
            corner += 37000

        if board[0][size - 1] == AI.myColor:
            corner += 37000

        if board[size - 1][0] == AI.myColor:
            corner += 37000

        if board[size - 1][size - 1] == AI.myColor:
            corner += 37000

        if board[0][0] == -AI.myColor:
            corner -= 37000

        if board[0][size - 1] == -AI.myColor:
            corner -= 37000

        if board[size - 1][0] == -AI.myColor:
            corner -= 37000

        if board[size - 1][size - 1] == -AI.myColor:
            corner -= 37000

        cornerX = 0
        if board[1][1] == AI.myColor and board[0][0] == 0:
            cornerX -= 19000

        if board[1][size - 2] == AI.myColor and board[0][size - 1] == 0:
            cornerX -= 19000

        if board[size - 2][1] == AI.myColor and board[size - 1][0] == 0:
            cornerX -= 19000

        if board[size - 2][size - 2] == AI.myColor and board[size - 1][size - 1] == 0:
            cornerX -= 19000

        if board[1][1] == -AI.myColor and board[0][0] == 0:
            cornerX += 19000

        if board[1][size - 2] == -AI.myColor and board[0][size - 1] == 0:
            cornerX += 19000

        if board[size - 2][1] == -AI.myColor and board[size - 1][0] == 0:
            cornerX += 19000

        if board[size - 2][size - 2] == -AI.myColor and board[size - 1][size - 1] == 0:
            cornerX += 19000


        edgeX = 0
        if board[0][1] == AI.myColor and board[0][0] == 0:
            edgeX -= 12505
        if board[0][1] == -AI.myColor and board[0][0] == 0:
            edgeX += 12505

        if board[0][size - 2] == AI.myColor and board[0][size-1] == 0:
            edgeX -= 12505
        if board[0][size - 2] == -AI.myColor and board[0][size-1] == 0:
            edgeX += 12505


        if board[1][0] == AI.myColor and board[0][0] == 0:
            edgeX -= 12505
        if board[1][0] == -AI.myColor and board[0][0] == 0:
            edgeX += 12505

        if board[1][size - 1] == AI.myColor and board[0][size-1] == 0:
            edgeX -= 12505
        if board[1][size - 1] == -AI.myColor and board[0][size-1] == 0:
            edgeX += 12505

        if board[size-2][0] == AI.myColor and board[size-1][0] == 0:
            edgeX -= 12505
        if board[size-2][0] == -AI.myColor and board[size-1][0] == 0:
            edgeX += 12505

        if board[size-1][1] == AI.myColor and board[size-1][0] == 0:
            edgeX -= 12505
        if board[size-1][1] == -AI.myColor and board[size-1][0] == 0:
            edgeX += 12505



        if board[size-2][size-1] == AI.myColor and board[size-1][size-1] == 0:
            edgeX -= 12505
        if board[size-2][size-1] == -AI.myColor and board[size-1][size-1] == 0:
            edgeX += 12505

        if board[size-1][size-2] == AI.myColor and board[size-1][size-1] == 0:
            edgeX -= 12505
        if board[size-1][size-2] == -AI.myColor and board[size-1][size-1] == 0:
            edgeX += 12505


        # calculate
        for i in range(size):
            for j in range(size):
                if board[i][j] == AI.myColor:
                    val += WEIGHTS[i][j]
                elif board[i][j] == -AI.myColor:
                    val -= WEIGHTS[i][j]

        state = AI.count(size, board)
        if state >= 58:
            for i in range(size):
                for j in range(size):
                    if board[i][j] == AI.myColor:
                        val += 1000
                    elif board[i][j] == -AI.myColor:
                        val -= 1000
        elif state <= 20:
            val += 2.6*corner
            val += 2.7*cornerX
            val += 2.3*edgeX
            val += 2.4*mobility
            val += edge*2300
            val -= outSide*7550
            val += stable*8810
        elif 20 < state <= 48:
            val += 2.3*corner
            val += 2.2*cornerX
            val += 2.1*edgeX
            val += 4.5*mobility
            val += edge*1800
            val -= outSide*7550
            val += inSide*2600
            val += stable*9070
        else:
            val += 1.6*corner
            val += 1.5*cornerX
            val += edgeX
            val += 2*mobility
            val -= outSide * 6875
            val += inSide * 1200
            val += stable * 6700

        return val

    @staticmethod
    def place(size, board, color, x, y):
        if x < 0 or x >= size or y < 0 or y >= size:
            return
        board[x][y] = color
        DIR = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        for d in range(8):
            i = x + DIR[d][0]
            j = y + DIR[d][1]
            while 0 <= i < size and 0 <= j < size and board[i][j] == -color:
                i += DIR[d][0]
                j += DIR[d][1]
            if 0 <= i < size and 0 <= j < size and board[i][j] == color:
                while True:
                    i -= DIR[d][0]
                    j -= DIR[d][1]
                    if (i, j) == (x, y):
                        break
                    board[i][j] = color

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
# cb = np.zeros((8, 8), dtype=np.int)
# cb[3][4], cb[4][3], cb[3][3], cb[4][4] = COLOR_BLACK, COLOR_BLACK, COLOR_WHITE, COLOR_WHITE
# ai = AI(8, COLOR_BLACK, 10)
#
# print(cb)
# print()
#
#
#
# for round in range(4, 64):
#     start = datetime.datetime.now()
#     ai.go(cb)
#     lens = len(ai.candidate_list)
#     print(ai.candidate_list)
#     cnt = AI.count(8,cb)
#     if lens != 0:
#         tx, ty = ai.candidate_list[lens-1]
#         AI.place(8, cb, ai.color, tx, ty)
#     print(cb)
#
#     ai.color = -ai.color
#     AI.myColor = -AI.myColor
#     end = datetime.datetime.now()
#     print(cnt, end-start)
#     print()


