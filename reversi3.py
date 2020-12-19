#####
# ! Something about transition table and zobrist hash is learned from http://www.soongsky.com/othello/computer/
######
import numpy as np
import random
import time

COLOR_BLACK = -1
COLOR_WHITE = 1
COLOR_NONE = 0


class Node_0124(object):
    def __init__(self, lock, depth, lower, upper, best_move):
        self.lock = lock
        self.lower = lower
        self.upper = upper
        self.best_move = best_move
        self.depth = depth


# don't change the class name
class AI(object):
    myColor = 0
    total_time = 0
    time_start = 0

    WEIGHTS = np.array([
        [220, -30, 110, 80, 80, 110, -30, 220],
        [-30, -80, -40, 10, 10, -40, -80, -30],
        [110, -40, 20, 20, 20, 20, -40, 110],
        [80, 13, 20, -29, -29, 20, 13, 80],
        [80, 13, 20, -29, -29, 20, 13, 80],
        [110, -40, 20, 20, 20, 20, -40, 110],
        [-30, -80, -40, 10, 10, -40, -80, -30],
        [220, -30, 110, 80, 80, 110, -30, 220]
    ])

    DIR = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    def __init__(self, chessboard_size, color, time_out):
        self.chessboard_size = chessboard_size
        self.color = color
        AI.myColor = color
        self.time_out = time_out
        AI.total_time = time_out
        self.candidate_list = []

    TABLE_SIZE = 1 << 16
    hash_table = [Node_0124(-1, -1, -1, -1, (-1, -1)) for j in range(TABLE_SIZE)]
    switch = 3803428188063167533
    black = [
        [5437766082965420031, 1604586077871189157, 5167034337698223703, 1879574480780201363, 8875003364533582782,
         2471433692035621528, 1814797639871649295, 401111281941334332],
        [1888989353073351375, 9349007780368593315, 1520653344881273507, 15131553424238542823, 10589945929206803489,
         1810380550432635992, 5224678496607662627, 6195654941553909527],
        [17563247847684269758, 9300476481210008215, 4254437910435036678, 14328559724165552299, 2810185164823779969,
         9067284540662615111, 11338504718529159786, 4956899636770778945],
        [5658547237689063494, 263351647634576482, 3571050432289826019, 10177322098146747152, 30238135032900566,
         10938255952335693937, 13830749389025968229, 13889670814295605726],
        [1575290572124372613, 16255825638786255721, 17559360801861166242, 2733292439550645079, 15281606825786484292,
         2332193130283695521, 10785563714297593874, 9401529854219930068],
        [9745580641037896103, 11982202779252069946, 629519581355525222, 7384798486994323040, 1240150288037109272,
         8045710612911807814, 4885540836415073818, 3040680703371910535],
        [4217937431914076728, 15026650864291844216, 8798818537061967991, 8424423204677230916, 14549635068113486640,
         7939535988104949548, 12434592127857881133, 13980252519975764755],
        [6427141398508560484, 9057379132413973786, 10182686583016085333, 15542603487921170427, 2354977815828635092,
         8461671775516397050, 12784883697746003596, 1348182986265921307]
    ]
    white = [
        [12143635282467254233, 16491715074970541999, 17099509932816580664, 5975442974156174047, 13118458175684501630,
         16783300213641809364, 6106432261787590927, 8010141361507630634],
        [1926802343522646825, 1714156373623098393, 16489358887392083785, 38628918900796358, 16000488279612615541,
         16343209308916178525, 17883725359683501631, 5389084520914615282],
        [6752074362832875299, 16171482834971538558, 2417241381039110138, 1167661141001918303, 6337959290699639268,
         2624562706536404911, 18411959167244069265, 13402562129105556990],
        [6693176841921451832, 13270582985316419444, 582641721467450746, 12416107716741037230, 9091612626346992547,
         10176944184119359937, 12628269689473231623, 13362854735666002486],
        [11373353880594290760, 10790602342262268273, 17211418556776443885, 14305208208808036244, 1680512849444371875,
         16898929056536798724, 14750254766906338931, 11948340510496342911],
        [8127870624675212272, 1506393091131479911, 5580971303755434179, 650256907124026539, 13763412837268679041,
         2800622950940172330, 16723245692237776744, 15263330577779016230],
        [977387522387671309, 17098272242521409650, 5278829944589829888, 10648409012897959153, 17164076276897027008,
         3200122664166998916, 9410991331502755491, 3219380947958999446],
        [6251818611145555072, 5404512854115369389, 13585736821607511688, 17665644905303250014, 8817701995760395215,
         15260295724792322049, 11925419328278010867, 16653492684804783742],
    ]

    def go(self, chessboard):
        AI.time_start = time.time()
        self.candidate_list.clear()
        self.candidate_list = AI.getMoves(self.chessboard_size, chessboard, self.color)
        self.candidate_list = AI.preProcess(self.candidate_list, self.chessboard_size, chessboard, self.color)
        AI.iterative_deepening(self.chessboard_size, 5, self.candidate_list, chessboard)


    @staticmethod
    def iterative_deepening(size, max_depth, canditate_list, chessboard):
        length = len(canditate_list)
        oldBoard = chessboard.copy()
        for dep in range(1, max_depth):
            cut = False
            maxV = -np.inf
            alpha = -np.inf
            beta = np.inf
            best_move = (-1, -1)
            hash_code = AI.get_hash_code(chessboard, AI.myColor)
            for i in range(length):
                tempX, tempY = canditate_list[i]
                AI.place(size, chessboard, AI.myColor, tempX, tempY)
                val = -AI.alphaBeta(size, chessboard, -beta, -alpha, -AI.myColor, dep)

                for x in range(size):
                    for y in range(size):
                        chessboard[x][y] = oldBoard[x][y]

                if val > alpha:
                    if val >= beta:
                        best_move = (tempX, tempY)
                        AI.update(hash_code, val, best_move, dep+1, alpha, beta)
                        cut = True
                        break
                    alpha = val
                if val > maxV:
                    maxV = val
                    best_move = (tempX, tempY)

            if not cut and length != 0:
                AI.update(hash_code, maxV, best_move, dep+1, alpha, beta)

            time_end = time.time()
            if best_move != (-1, -1) and (time_end - AI.time_start) <= AI.total_time - 0.1:
                canditate_list.append(best_move)
            else:
                break


    @staticmethod
    def preProcess(candidate_list, size, board, color):
        score = []
        oldBoard = board.copy()
        for m in candidate_list:
            tempX, tempY = m
            AI.place(size, board, color, tempX, tempY)
            temp = AI.evaluate(size, board)
            if m == (0, 0) or m == (0, size-1) or m == (size-1, 0) or m == (size-1, size-1):
                temp += 50000
            if m == (1, 1) or m == (1, size-2) or m == (size-2, 1) or m == (size-2, size-2):
                temp -= 50000
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
            for direction in range(8):
                dx = AI.DIR[direction][0]
                dy = AI.DIR[direction][1]
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
        opp_num = my_num = 0
        my_in = opp_in = 0
        my_edge = opp_edge = 0
        edge = 0

        # number
        for i in range(size):
            for j in range(size):
                if board[i][j] == AI.myColor:
                    my_num += 1
                elif board[i][j] == -AI.myColor:
                    opp_num += 1
        if my_num > opp_num:
            number = (100 * my_num) / (my_num + opp_num + 0.1)
        else:
            number = -(100 * opp_num) / (my_num + opp_num + 0.1)

        # edge
        my_edge = AI.get_edge(board, AI.myColor, size)
        opp_edge = AI.get_edge(board, -AI.myColor, size)
        if my_edge > opp_edge:
            edge = (100 * my_edge) / (my_edge + opp_edge + 0.1)
        else:
            edge = -(100 * opp_edge) / (my_edge + opp_edge + 0.1)


        # mobility
        for i in range(size):
            for j in range(size):
                if board[i][j] == 0:
                    if AI.isValid(size, board, -AI.myColor, i, j):
                        opp_mob += 1
                    if AI.isValid(size, board, AI.myColor, i, j):
                        my_mob += 1

        if my_mob > opp_mob:
            mobility = (100 * my_mob) / (my_mob + opp_mob + 0.1)
        else:
            mobility = -(100 * opp_mob) / (my_mob + opp_mob + 0.1)

        # outside
        for x in range(size):
            for y in range(size):
                if board[x][y] != 0:
                    for d in range(8):
                        tempX = x + AI.DIR[d][0]
                        tempY = y + AI.DIR[d][1]
                        if 0 <= tempX < size and 0 <= tempY < size and board[tempX][tempY] == 0:
                            if board[x][y] == AI.myColor:
                                my_out += 1
                            else:
                                opp_out += 1
        if my_out > opp_out:
            outside = -(100 * my_out) / (my_out + opp_out + 0.1)
        else:
            outside = (100 * opp_out) / (my_out + opp_out + 0.1)

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
                xLocation += 0.29
            elif board[0][1] == -AI.myColor:
                xLocation -= 0.29

            if board[1][0] == AI.myColor:
                xLocation += 0.29
            elif board[1][0] == -AI.myColor:
                xLocation -= 0.29

            if board[1][1] == AI.myColor:
                xLocation += 1
            elif board[1][1] == -AI.myColor:
                xLocation -= 1

        if board[size - 1][size - 1] == 0:
            if board[size - 1][size - 2] == AI.myColor:
                xLocation += 0.29
            elif board[size - 1][size - 2] == -AI.myColor:
                xLocation -= 0.29

            if board[size - 2][size - 1] == AI.myColor:
                xLocation += 0.29
            elif board[size - 2][size - 1] == -AI.myColor:
                xLocation -= 0.29

            if board[size - 2][size - 2] == AI.myColor:
                xLocation += 1
            elif board[size - 2][size - 2] == -AI.myColor:
                xLocation -= 1

        if board[0][size - 1] == 0:
            if board[0][size - 2] == AI.myColor:
                xLocation += 0.29
            elif board[0][size - 2] == -AI.myColor:
                xLocation -= 0.29

            if board[1][size - 1] == AI.myColor:
                xLocation += 0.29
            elif board[1][size - 1] == -AI.myColor:
                xLocation -= 0.29

            if board[1][size - 2] == AI.myColor:
                xLocation += 1
            elif board[1][size - 2] == -AI.myColor:
                xLocation -= 1

        if board[size - 1][0] == 0:
            if board[size - 1][1] == AI.myColor:
                xLocation += 0.29
            elif board[size - 1][1] == -AI.myColor:
                xLocation -= 0.29

            if board[size - 2][0] == AI.myColor:
                xLocation += 0.29
            elif board[size - 2][0] == -AI.myColor:
                xLocation -= 0.29

            if board[size - 2][1] == AI.myColor:
                xLocation += 1
            elif board[size - 2][1] == -AI.myColor:
                xLocation -= 1

        # stable
        stable = AI.get_stable(board, AI.myColor, size) - AI.get_stable(board, -AI.myColor, size)

        # inside
        for x in range(size):
            for y in range(size):
                if board[x][y] != 0:
                    flag = True
                    for d in range(8):
                        tempX = x + AI.DIR[d][0]
                        tempY = y + AI.DIR[d][1]
                        if 0 <= tempX < size and 0 <= tempY < size and board[tempX][tempY] == 0:
                            flag = False
                            break
                    if flag:
                        if board[x][y] == AI.myColor:
                            my_in += 1
                        else:
                            opp_in += 1
        if my_in > opp_in:
            inside = (100 * my_in) / (my_in + opp_in + 0.1)
        else:
            inside = -(100 * opp_in) / (my_in + opp_in + 0.1)

        # map weight
        for i in range(size):
            for j in range(size):
                if board[i][j] == AI.myColor:
                    map_val += AI.WEIGHTS[i][j]
                elif board[i][j] == -AI.myColor:
                    map_val -= AI.WEIGHTS[i][j]

        value = 0
        state = AI.count(size, board)
        if state > 61:
            value = 100 * number
        elif 0 < state <= 15:
            value = 22890 * corner - 15677 * xLocation + 86 * mobility + 67 * stable + 78 * outside + map_val
        elif 15 < state <= 18:
            value = 23890 * corner - 15777 * xLocation + 98 * mobility + 100 * stable + 76 * outside + map_val + 22 * edge
        elif 18 < state <= 21:
            value = 24890 * corner - 15777 * xLocation + 108 * mobility + 200 * stable + 4 * inside + 76 * outside + map_val + 24 * edge
        elif 21 < state <= 24:
            value = 5 * number + 25890 * corner - 15377 * xLocation + 116 * mobility + 300 * stable + 5 * inside + 72 * outside + map_val + 22 * edge
        elif 24 < state <= 37:
            value = 7 * number + 31350 * corner - 15045 * xLocation + 124 * mobility + 1000 * stable + 6 * inside + 70 * outside + map_val + 22 * edge
        elif 37 < state <= 40:
            value = 7 * number + 27350 * corner - 15877 * xLocation + 124 * mobility + 1500 * stable + 7 * inside + 66 * outside + map_val + 16 * edge
        elif 40 < state <= 49:
            value = 8 * number + 24350 * corner - 15877 * xLocation + 124 * mobility + 2000 * stable + 60 * outside + map_val
        else:
            value = 9 * number + 23060 * corner - 15877 * xLocation + 120 * mobility + 2250 * stable + 56 * outside + map_val

        return value

    @staticmethod
    def get_edge(board, color, size):
        edge = 0
        for i in range(3, size - 2):
            if board[i][0] == color:
                edge += 1
            if board[i][size - 1] == color:
                edge += 1
            if board[size - 1][i] == color:
                edge += 1
            if board[0][i] == color:
                edge += 1
        return edge

    @staticmethod
    def get_stable(board, color, size):
        is_stable = np.full((size, size), False)
        row_judge = np.full(size, True)
        col_judge = np.full(size, True)
        dig_judge1 = np.full(15, True)
        dig_judge2 = np.full(15, True)
        cnt = 0

        # corner and pieces next to it
        for i in range(size):
            for j in range(size):
                if board[i][j] == 0:
                    row_judge[i] = False
                    col_judge[j] = False
                    dig_judge1[i+j] = False
                    dig_judge2[7 - i + j] = False

        if board[0][0] == color:
            is_stable[0][0] = True
            for j in range(1, size - 1):
                if board[0][j] == color:
                    is_stable[0][j] = True
                else:
                    break
            for i in range(1, size - 1):
                if board[i][0] == color:
                    is_stable[i][0] = True
                else:
                    break

        if board[size - 1][0] == color:
            is_stable[size - 1][0] = True
            for i in range(size - 2, 0, -1):
                if board[i][0] == color:
                    is_stable[i][0] = True
                else:
                    break
            for j in range(1, size - 1):
                if board[size - 1][j] == color:
                    is_stable[size - 1][j] = True
                else:
                    break

        if board[0][size - 1] == color:
            is_stable[0][size - 1] = True
            for i in range(1, size - 1):
                if board[i][size - 1] == color:
                    is_stable[i][size - 1] = True
                else:
                    break
            for j in range(size - 2, 0, -1):
                if board[0][j] == color:
                    is_stable[0][j] = True
                else:
                    break

        if board[size - 1][size - 1] == color:
            is_stable[size - 1][size - 1] = True
            for i in range(size - 2, 0, -1):
                if board[i][size - 1] == color:
                    is_stable[i][size - 1] = True
                else:
                    break
            for j in range(size - 2, 0, -1):
                if board[size - 1][j] == color:
                    is_stable[size - 1][j] = True
                else:
                    break

        # full edge
        for i in [0, size - 1]:
            for j in range(size):
                if row_judge[i] and board[i][j] == color:
                    is_stable[i][j] = True
            for j in range(size):
                if col_judge[i] and board[j][i] == color:
                    is_stable[j][i] = True

        # no blanks in all directions
        for i in range(1, size - 1):
            for j in range(1, size - 1):
                if board[i][j] == color and row_judge[i] and col_judge[j] and dig_judge1[i + j] and dig_judge2[7 - i + j]:
                    is_stable[i][j] = True

        for i in range(size):
            for j in range(size):
                if is_stable[i][j]:
                    cnt += 1

        return cnt

    @staticmethod
    def place(size, board, color, x, y):
        if x < 0 or x >= size or y < 0 or y >= size:
            return
        board[x][y] = color
        for d in range(8):
            tempX = x + AI.DIR[d][0]
            tempY = y + AI.DIR[d][1]
            while 0 <= tempX < size and 0 <= tempY < size and board[tempX][tempY] == -color:
                tempX += AI.DIR[d][0]
                tempY += AI.DIR[d][1]
            if 0 <= tempX < size and 0 <= tempY < size and board[tempX][tempY] == color:
                while True:
                    tempX -= AI.DIR[d][0]
                    tempY -= AI.DIR[d][1]
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

        hash_code = AI.get_hash_code(board, color)
        node = AI.get_hash(hash_code, depth)
        if node is not None:
            if node.lower > alpha:
                alpha = node.lower
                if alpha >= beta:
                    return alpha
            if node.upper < beta:
                beta = node.upper
                if beta <= alpha:
                    return beta

        moves = AI.getMoves(size, board, color)
        moves = AI.preProcess(moves, size, board, color)
        oldBoard = board.copy()
        best_move = (-1, -1)
        for m in moves:
            tempX, tempY = m
            AI.place(size, board, color, tempX, tempY)
            val = -AI.alphaBeta(size, board, -beta, -alpha, -color, depth-1)

            for i in range(size):
                for j in range(size):
                    board[i][j] = oldBoard[i][j]

            if val > alpha:
                if val >= beta:
                    best_move = m
                    AI.update(hash_code, val, best_move, depth, alpha, beta)
                    return val
                alpha = val
            if val > maxV:
                maxV = val
                best_move = m

        AI.update(hash_code, maxV, best_move, depth, alpha, beta)
        return maxV

    @staticmethod
    def update(hash_code, best_value, best_move, depth, alpha, beta):
        if best_value >= beta:
            AI.update_hash(hash_code, best_value, np.inf, best_move, depth)
        elif best_value <= alpha:
            AI.update_hash(hash_code, -np.inf, best_value, best_move, depth)
        else:
            AI.update_hash(hash_code, best_value, best_value, best_move, depth)

    @staticmethod
    def get_hash_code(board, color):
        hash_code = 0
        for i in range(8):
            for j in range(8):
                if board[i][j] == -1:
                    hash_code ^= AI.black[i][j]
                elif board[i][j] == 1:
                    hash_code ^= AI.white[i][j]
        if color == 1:
            hash_code ^= AI.switch
        return hash_code

    @staticmethod
    def update_hash(hashcode, lower, upper, best_move, depth):
        index = hashcode & (AI.TABLE_SIZE - 1)
        if hashcode == AI.hash_table[index].lock and depth == AI.hash_table[index].depth:
            if lower > AI.hash_table[index].lower:
                AI.hash_table[index].lower = lower
                AI.hash_table[index].best_move = best_move
            if upper > AI.hash_table[index].upper:
                AI.hash_table[index].upper = upper
        elif depth < AI.hash_table[index].depth:
            AI.hash_table[index].lock = hashcode
            AI.hash_table[index].depth = depth
            AI.hash_table[index].best_move = best_move
            AI.hash_table[index].upper = upper
            AI.hash_table[index].lower = lower
        else:
            AI.hash_table[index].lock = hashcode
            AI.hash_table[index].depth = depth
            AI.hash_table[index].best_move = best_move
            AI.hash_table[index].upper = upper
            AI.hash_table[index].lower = lower


    @staticmethod
    def get_hash(hashcode, depth):
        index = hashcode & (AI.TABLE_SIZE - 1)
        if AI.hash_table[index].lock == -1:
            return None
        elif AI.hash_table[index].lock == hashcode and AI.hash_table[index].depth == depth:
            return AI.hash_table[index]
        else:
            return None


# import datetime
# # cd = [[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0], [0, 1, 1, 1, 1, 1, 0, 0], [0, 0, 1, 1, 1, -1, -1, 0], [0, 0, 1, -1, -1, -1, -1, -1], [0, 0, 0, -1, -1, -1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]]
# # cb = np.zeros((8, 8), dtype=np.int)
# # for i in range(8):
# #     for j in range(8):
# #         cb[i][j] = cd[i][j]
# # # cb[3][4], cb[4][3], cb[3][3], cb[4][4] = COLOR_BLACK, COLOR_BLACK, COLOR_WHITE, COLOR_WHITE
# # print(AI.count(8, cb))
# # ai = AI(8, -1, 5)
# # print(cb)
# # ai.go(cb)
# # print(ai.candidate_list)




# import datetime
# cb = np.zeros((8, 8), dtype=np.int)
# cb[3][4], cb[4][3], cb[3][3], cb[4][4] = COLOR_BLACK, COLOR_BLACK, COLOR_WHITE, COLOR_WHITE
# ai = AI(8, COLOR_BLACK, 5)
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
