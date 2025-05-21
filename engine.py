import chess
import chess.polyglot
import time

# -----------------------------------
# Transposition Table Entry
# -----------------------------------
class TTEntry:
    def __init__(self, depth, score, flag, best_move):
        self.depth = depth
        self.score = score
        self.flag = flag  # EXACT, LOWERBOUND, UPPERBOUND
        self.best_move = best_move


# -----------------------------------
# Main Engine Class
# -----------------------------------
class ChessEngine:
    def __init__(self):
        self.tt = {}
        self.node_count = 0
        self.start_time = 0
        self.time_limit = 1.0  # seconds
        self.INF = float('inf')
        self.MAX_DEPTH = 64
        self.piece_values = {
            chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
            chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0
        }
        self.piece_square_tables = {
            chess.PAWN: [
                0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  ,
                0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  ,
                0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  ,
                0  , 0  , 0  , 0  , 0  , 0  , 0  , 0  ,
                5  , 10 , 10 , -20, -20, -10, -10, -5 ,
                -5 , -10, -10, -20, -20, -10, -10, -5 ,
                5  , 10 , 10 , -20, -20, -10, -10, -5 ,
            ],
            chess.KNIGHT: [
                -50, -40, -30, -30, -30, -30, -40, -50,
                -40, -20, 0  , 0  , 0  , 0  , -20, -40,
                -30, 0  , 10 , 15 , 15 , 10 , 0  , -30,
                -30, 5  , 15 , 20 , 20 , 15 , 5  , -30,
                -30, 0  , 15 , 20 , 20 , 15 , 0  , -30,
                -30, 5  , 10 , 15 , 15 , 10 , 5  , -30,
                -40, -20, 0  , 0  , 0  , 0  , -20, -40,
                -50, -40, -30, -30, -30, -30, -40, -50
            ],
            chess.BISHOP: [
                -20, -10, -10, -10, -10, -10, -10, -20,
                -10, 0  , 0  , 0  , 0  , 0  , 0  , -10,
                -10, 0  , 5  , 10 , 10 , 5  , 0  , -10,
                -10, 5  , 5  , 10 , 10 , 5  , 5  , -10,
                -10, 0  , 5  , 10 , 10 , 5  , 0  , -10,
                -10, 0  , 0  , 0  , 0  , 0  , -20, -20,
                -20, -20, -20, -20, -20, -20, -20, -20
            ],
            chess.ROOK: [
                
            ]
        }

    # -----------------------------------
    # Evaluation Function
    # -----------------------------------
    def evaluate(self, board):
        self.count_material_evaluation(board)

    # -----------------------------------
    # Iterative Deepening Search
    # -----------------------------------
    def search(self, board, time_limit=1.0) -> (chess.Move, int, int):
        self.start_time = time.time()
        self.time_limit = time_limit
        best_move = None
        try:
            for depth in range(1, self.MAX_DEPTH + 1):
                self.node_count = 0
                score = self.alpha_beta(board, depth, -self.INF, self.INF, is_pv_node=True, ply=0)
                hash_key = chess.polyglot.zobrist_hash(board)
                best_move = self.tt[hash_key].best_move if hash_key in self.tt else best_move
                if time.time() - self.start_time > self.time_limit:
                    break
        except TimeoutError:
            pass

        if best_move is None:
            raise TimeoutError("No best move found within time limit.")

        return best_move, score, depth, self.node_count

    # -----------------------------------
    # Zobrist Hashing using python-chess
    # -----------------------------------
    @staticmethod
    def hash_board(board):
        return chess.polyglot.zobrist_hash(board)

    @staticmethod
    def is_drawn(board) -> chess.Outcome | False:
        if board.is_stalemate() or board.is_insufficient_material() or board.is_fivefold_repetition() or board.is_seventyfive_moves():
            return board.outcome()
        return False

    # -----------------------------------
    # Move Ordering Heuristic
    # -----------------------------------
    def order_moves(self, board, moves, tt_move):
        def score(move):
            if move == tt_move:
                return 10000
            if board.is_capture(move):
                # Higher score for captures with a piece less valuable than the victim
                victim = board.piece_at(move.to_square)
                attacker = board.piece_at(move.from_square)
                if victim and attacker:
                    return 1000 + self.piece_values[victim.piece_type] - self.piece_values[attacker.piece_type]
            return 0
        return sorted(moves, key=score, reverse=True)

    # -----------------------------------
    # Alpha-Beta with Extensions, LMR, and TT
    # -----------------------------------
    def alpha_beta(self, board, depth, alpha, beta, is_pv_node, ply):
        self.node_count += 1
        if time.time() - self.start_time > self.time_limit:
            raise TimeoutError

        hash_key = self.hash_board(board)
        if hash_key in self.tt:
            entry = self.tt[hash_key]
            if entry.depth >= depth:
                if entry.flag == 'EXACT':
                    return entry.score
                elif entry.flag == 'LOWERBOUND':
                    alpha = max(alpha, entry.score)
                elif entry.flag == 'UPPERBOUND':
                    beta = min(beta, entry.score)
                if alpha >= beta:
                    return entry.score

        if depth == 0:
            if board.is_check() or any(board.is_capture(m) for m in board.legal_moves):
                # Extend in check or capture at leaf
                depth += 1
            else:
                return self.evaluate(board)

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return self.evaluate(board)

        best_score = -self.INF
        best_move = None

        tt_move = self.tt[hash_key].best_move if hash_key in self.tt else None
        ordered_moves = self.order_moves(board, legal_moves, tt_move)

        for i, move in enumerate(ordered_moves):
            board.push(move)

            is_capture = board.is_capture(move)
            is_check = board.is_check()
            reduction = 1 if depth >= 3 and i >= 3 and not is_capture and not is_check and not is_pv_node else 0
            reduced_depth = depth - 1 - reduction

            try:
                if reduced_depth <= 0:
                    score = -self.evaluate(board)
                else:
                    score = -self.alpha_beta(board, reduced_depth, -beta, -alpha, is_pv_node=False, ply=ply + 1)
            except TimeoutError:
                board.pop()
                raise

            board.pop()

            if score > best_score:
                best_score = score
                best_move = move
                if score > alpha:
                    alpha = score
                if alpha >= beta:
                    break

        # Save to transposition table
        flag = 'EXACT' if alpha < beta else 'LOWERBOUND' if best_score >= beta else 'UPPERBOUND'
        self.tt[hash_key] = TTEntry(depth, best_score, flag, best_move)

        return best_score

    def count_material_evaluation(self, board):
        if board.is_checkmate():
            return -self.INF if board.turn else self.INF  # board.turn is True if white's turn
        if self.is_drawn(board):
            return 0

        score = 0
        for piece_type in self.piece_values:
            score += len(board.pieces(piece_type, chess.WHITE)) * self.piece_values[piece_type]
            score -= len(board.pieces(piece_type, chess.BLACK)) * self.piece_values[piece_type]
        return score

    def piece_square_value_evaluation(self, board):
        # Placeholder for piece-square table evaluation
        # This function can be implemented with specific piece-square tables for each piece type
        return 0

