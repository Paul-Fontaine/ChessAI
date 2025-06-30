import chess
import chess.polyglot
import time
from board_helper_functions import *
from nnue import KPNNUE

KING_ENDGAME = 7
MATE_SCORE = 30000


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
        self.q_count = 0
        self.q_nodes = 0
        self.tt_use_count = 0
        self.start_time = 0
        self.time_limit = 1.0  # seconds
        self.MAX_DEPTH = 6
        self.killer_moves = [[None, None] for _ in range(self.MAX_DEPTH+5)]  # ply-indexed, 2 moves stored per depth
        self.history_heuristic = {}  # (from_square, to_square) -> score
        self.piece_value = {
            chess.PAWN: 100, chess.KNIGHT: 320, chess.BISHOP: 330,
            chess.ROOK: 500, chess.QUEEN: 900, chess.KING: 0
        }
        # White perspective; black will flip ranks.
        self.PST = {
            chess.PAWN: [
                 0,   0,   0,   0,   0,   0,   0,   0,
                 5,  10,  10, -20, -20,  10,  10,   5,
                 5,  -5, -10,   0,   0, -10,  -5,   5,
                 0,   0,   0,  20,  20,   0,   0,   0,
                 5,   5,  10,  25,  25,  10,   5,   5,
                10,  10,  20,  30,  30,  20,  10,  10,
                50,  50,  50,  50,  50,  50,  50,  50,
                 0,   0,   0,   0,   0,   0,   0,   0
            ],
            chess.KNIGHT: [
                -50, -40, -30, -30, -30, -30, -40, -50,
                -40, -20,   0,   5,   5,   0, -20, -40,
                -30,   5,  10,  15,  15,  10,   5, -30,
                -30,   0,  15,  20,  20,  15,   0, -30,
                -30,   5,  15,  20,  20,  15,   5, -30,
                -30,   0,  10,  15,  15,  10,   0, -30,
                -40, -20,   0,   0,   0,   0, -20, -40,
                -50, -40, -30, -30, -30, -30, -40, -50
            ],
            chess.BISHOP: [
                -20, -10, -10, -10, -10, -10, -10, -20,
                -10,   5,   0,   0,   0,   0,   5, -10,
                -10,  10,  10,  10,  10,  10,  10, -10,
                -10,   0,  10,  10,  10,  10,   0, -10,
                -10,   5,   5,  10,  10,   5,   5, -10,
                -10,   0,   5,  10,  10,   5,   0, -10,
                -10,   0,   0,   0,   0,   0,   0, -10,
                -20, -10, -10, -10, -10, -10, -10, -20
            ],
            chess.ROOK: [
                 0,   0,   0,   0,   0,   0,   0,   0,
                 5,  10,  10,  10,  10,  10,  10,   5,
                -5,   0,   0,   0,   0,   0,   0,  -5,
                -5,   0,   0,   0,   0,   0,   0,  -5,
                -5,   0,   0,   0,   0,   0,   0,  -5,
                -5,   0,   0,   0,   0,   0,   0,  -5,
                -5,   0,   0,   0,   0,   0,   0,  -5,
                 0,   0,   0,   5,   5,   0,   0,   0
            ],
            chess.QUEEN: [
                -20, -10, -10,  -5,  -5, -10, -10, -20,
                -10,   0,   0,   0,   0,   0,   0, -10,
                -10,   0,   5,   5,   5,   5,   0, -10,
                 -5,   0,   5,   5,   5,   5,   0,  -5,
                  0,   0,   5,   5,   5,   5,   0,  -5,
                -10,   5,   5,   5,   5,   5,   0, -10,
                -10,   0,   5,   0,   0,   0,   0, -10,
                -20, -10, -10,  -5,  -5, -10, -10, -20
            ],
            chess.KING: [
                -30, -40, -40, -50, -50, -40, -40, -30,
                -30, -40, -40, -50, -50, -40, -40, -30,
                -30, -40, -40, -50, -50, -40, -40, -30,
                -30, -40, -40, -50, -50, -40, -40, -30,
                -20, -30, -30, -40, -40, -30, -30, -20,
                -10, -20, -20, -20, -20, -20, -20, -10,
                 20,  20,   0,   0,   0,   0,  20,  20,
                 20,  30,  10,   0,   0,  10,  30,  20
            ],
            KING_ENDGAME: [
                -50, -40, -30, -20, -20, -30, -40, -50,
                -30, -20, -10,   0,   0, -10, -20, -30,
                -30, -10,  20,  30,  30,  20, -10, -30,
                -30, -10,  30,  40,  40,  30, -10, -30,
                -30, -10,  30,  40,  40,  30, -10, -30,
                -30, -10,  20,  30,  30,  20, -10, -30,
                -30, -30,   0,   0,   0,   0, -30, -30,
                -50, -30, -30, -30, -30, -30, -30, -50
            ]
        }
        self.openning_book = chess.polyglot.open_reader('komodo.bin')
        self.nnue = KPNNUE()

    def count_material(self, board):
        score = 0
        for piece_type in self.piece_value:
            score += len(board.pieces(piece_type, chess.WHITE)) * self.piece_value[piece_type]
            score -= len(board.pieces(piece_type, chess.BLACK)) * self.piece_value[piece_type]
        return score

    def evaluate_with_PST(self, board):
        score = 0
        for sq in chess.SQUARES:
            piece = board.piece_at(sq)
            if piece:
                base = self.piece_value[piece.piece_type]
                pst = self.PST[piece.piece_type]
                if is_endgame(board, self.piece_value):
                    pst = self.PST[KING_ENDGAME]
                idx = sq if piece.color == chess.WHITE else chess.square_mirror(sq)
                value = base + pst[idx]
                score += value if piece.color == chess.WHITE else -value
        return score

    def evaluate_with_nnue(self):
        return self.nnue.get_evaluation_incremental()

    def evaluate(self, board, ply=0):
        if board.is_checkmate():
            mate_eval = MATE_SCORE - ply
            return -mate_eval if board.turn else mate_eval  # board.turn is True if white's turn
        score = self.evaluate_with_PST(board)
        return score

    def get_book_move(self, board):
        try:
            entry = self.openning_book.find(board)
            return entry.move
        except IndexError:
            return None

    def order_moves(self, board, moves, tt_move=None, ply=0):
        def move_score(move):
            # PV move first
            if tt_move and move == tt_move:
                return 10000
            # Captures: MVV-LVA
            if board.is_capture(move):
                victim = board.piece_at(move.to_square)
                attacker = board.piece_at(move.from_square)
                if victim and attacker:
                    return 9000 + 10 * victim.piece_type - attacker.piece_type
                return 8000  # assume good capture if unsure
            # Killer moves
            if move in self.killer_moves[ply]:
                return 7000
            # History heuristic
            key = (move.from_square, move.to_square)
            return self.history_heuristic.get(key, 0)  # Default to 0 if move not found in history

        return sorted(moves, key=move_score, reverse=True)

    def quiescence(self, board, alpha, beta, ply, q_depth=0):
        """
        Quiescence search: extends search past depth 0 to avoid evaluating unstable positions.
        Only considers 'noisy' moves: captures, checks, and promotions.
        Returns:
            - score: best evaluation found in quiet positions
        """

        if q_depth > 0:
            self.q_nodes += 1
        # Safety: avoid infinite quiescence recursion
        if q_depth >= 3:
            return self.evaluate(board, ply)

        # Base score if we do nothing from here (stand-pat score)
        stand_pat = self.evaluate(board, ply)

        # Beta cutoff: we already have a better move earlier in the tree
        if stand_pat >= beta:
            return beta

        # Raise alpha if this is the best we've seen so far
        if alpha < stand_pat:
            alpha = stand_pat

        # Loop over all legal moves
        interesting_moves = [move for move in board.legal_moves if is_interesting(board, move)]
        ordered_moves = self.order_moves(board, interesting_moves)
        for move in ordered_moves:
            board.push(move)
            self.nnue.update_accumulator(board)
            score = -self.quiescence(board, -beta, -alpha, ply+1, q_depth + 1)
            board.pop()
            self.nnue.update_accumulator(board)

            # Prune if score is too high
            if score >= beta:
                return beta

            # Update best score
            if score > alpha:
                alpha = score

        return alpha

    def alpha_beta(self, board, depth, alpha, beta, is_pv_node, ply):
        if time.time() - self.start_time > self.time_limit:
            raise TimeoutError("Time limit exceeded during search.")

        self.node_count += 1
        hash_key = hash_board(board)

        # Check for draw conditions
        if is_draw(board):
            self.tt[hash_key] = TTEntry(depth, 0, 'EXACT', None)
            return 0, []
        if is_claimable_draw(board):
            if 0 >= beta:
                self.tt[hash_key] = TTEntry(depth, 0, 'LOWERBOUND', None)
                return 0, []
            alpha = max(alpha, 0)

        if depth == 0:
            return self.quiescence(board, alpha, beta, ply+1), []

        tt_move = None
        if hash_key in self.tt:
            self.tt_use_count += 1
            entry = self.tt[hash_key]
            tt_move = entry.best_move
            if entry.depth >= depth:
                if entry.flag == 'EXACT':
                    return entry.score, [tt_move] if tt_move else []
                elif entry.flag == 'LOWERBOUND':
                    alpha = max(alpha, entry.score)
                elif entry.flag == 'UPPERBOUND':
                    beta = min(beta, entry.score)
                if alpha >= beta:
                    return entry.score, [tt_move] if tt_move else []

        # --- Null Move Pruning ---
        if (
                not is_pv_node
                and depth >= 3
                and ply > 0
                and not board.is_check()
                and not is_endgame(board, self.piece_value)
        ):
            R = 2  # Reduction amount
            board.push(chess.Move.null())
            self.nnue.update_accumulator(board)
            try:
                null_score, _ = self.alpha_beta(board, depth - 1 - R, -beta, -beta + 1, False, ply + 1)
                null_score = -null_score
            except TimeoutError:
                board.pop()
                raise
            board.pop()
            self.nnue.update_accumulator(board)

            if null_score >= beta:
                return null_score, []  # Prune the branch

        # --- Razor Pruning ---
        if (
                depth == 1
                and not is_pv_node
                and not board.is_check()
                and not is_endgame(board, self.piece_value)
        ):
            static_eval = self.evaluate(board, ply)
            RAZOR_MARGIN = 300
            if static_eval + RAZOR_MARGIN < alpha:
                score = self.quiescence(board, alpha, beta, ply)
                return score, []

        legal_moves = list(board.legal_moves)
        if not legal_moves:
            score = self.evaluate(board, ply)
            self.tt[hash_key] = TTEntry(depth, score, 'EXACT', None)
            return score, []

        ordered_moves = self.order_moves(board, legal_moves, tt_move)

        best_score = -MATE_SCORE
        best_move = None
        best_line = []

        for i, move in enumerate(ordered_moves):
            # --- Futility Pruning ---
            if (
                    depth == 1
                    and not is_pv_node
                    and is_quiet(board, move)
                    and not is_endgame(board, self.piece_value)
            ):
                static_eval = self.evaluate(board, ply)
                FUTILITY_MARGIN = 150
                if static_eval + FUTILITY_MARGIN <= alpha:
                    continue  # Skip this move as it probably won't raise alpha

            board.push(move)
            self.nnue.update_accumulator(board)
            try:
                child_is_pv = is_pv_node and (i == 0)

                # --- Late Move Reduction ---
                if (
                        depth >= 3
                        and not child_is_pv
                        and is_quiet(board, move)
                        and i >= 3  # not among the first 3 moves
                        and not is_endgame(board, self.piece_value)
                ):
                    # Reduced-depth search first
                    R = 1 if i < 5 else 2  # Reduce depth by 1 or 2 ply based on move index
                    reduced_depth = depth - R
                    red_score, _ = self.alpha_beta(board, reduced_depth, -alpha - 1, -alpha, False, ply + 1)
                    red_score = -red_score

                    # If it fails high, do a full-depth re-search
                    if red_score > alpha:
                        score, line = self.alpha_beta(board, depth - 1, -beta, -alpha, child_is_pv, ply + 1)
                        score = -score
                    else:
                        score = red_score
                        line = []  # We skip storing line in reduced node
                else:
                    score, line = self.alpha_beta(board, depth - 1, -beta, -alpha, child_is_pv, ply + 1)
                    score = -score

            except TimeoutError:
                board.pop()
                raise
            board.pop()
            self.nnue.update_accumulator(board)

            if score > best_score:
                best_score = score
                best_move = move
                best_line = [move] + line

                alpha = max(alpha, score)
                if alpha >= beta:
                    if not board.is_capture(move):
                        killers = self.killer_moves[ply]
                        if move not in killers:
                            self.killer_moves[ply][1] = killers[0]
                            self.killer_moves[ply][0] = move
                        key = (move.from_square, move.to_square)
                        self.history_heuristic[key] = self.history_heuristic.get(key, 0) + depth * depth

                    self.tt[hash_key] = TTEntry(depth, best_score, 'LOWERBOUND', move)
                    return best_score, best_line

        flag = 'EXACT' if alpha < beta else 'LOWERBOUND' if best_score >= beta else 'UPPERBOUND'
        self.tt[hash_key] = TTEntry(depth, best_score, flag, best_move)

        return best_score, best_line

    # Iterative Deepening Search
    def search(self, board, time_limit=1.0) -> chess.Move:
        book_move = self.get_book_move(board)
        if book_move:
            print(f"Book move: {board.san(book_move)}")
            return book_move  # No search needed

        self.node_count = 0
        self.q_count = 0
        self.q_nodes = 0
        self.tt_use_count = 0
        self.start_time = time.time()
        self.time_limit = time_limit
        best_move = None
        self.nnue.init_accumulator(board)

        try:
            for depth in range(1, self.MAX_DEPTH + 1):
                score, best_line = self.alpha_beta(board, depth, -MATE_SCORE, MATE_SCORE, is_pv_node=True, ply=0)
                if best_line:
                    best_move = best_line[0]
                print(f" depth : {depth} ; node_count: {self.node_count} ; q nodes: {self.q_nodes}")

        except TimeoutError:
            pass

        if best_move is None:
            raise TimeoutError("No best move found within time limit.")

        duration = time.time() - self.start_time
        speed = self.node_count / duration
        print_best_line_san(best_line, board.copy())
        print(f"move: {board.san(best_move)} ; score: {-score / 100:.1f} \n"
              f"depth reached: {depth} ; node_count: {self.node_count} ; q nodes: {self.q_nodes} \n"
              f"tt_use_count: {self.tt_use_count} ; time: {duration:.2f} seconds ; speed: {speed:.0f} nodes/s")

        return best_move
