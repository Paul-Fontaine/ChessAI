import chess
import chess.polyglot


def is_promotion(board, move):
    piece = board.piece_at(move.from_square)
    if not piece or piece.piece_type != chess.PAWN:
        return False
    if piece.color == chess.WHITE and chess.square_rank(move.to_square) == 7:
        return True
    if piece.color == chess.BLACK and chess.square_rank(move.to_square) == 0:
        return True
    return False


def is_interesting(board, move):
    if is_promotion(board, move):
        return True
    if board.gives_check(move):
        return True
    if board.is_capture(move):
        # Only allow capture if victim is more valuable than attacker
        attacker = board.piece_at(move.from_square)
        victim = board.piece_at(move.to_square)
        if attacker and victim and victim.piece_type >= attacker.piece_type:
            return True
    return False


def hash_board(board):
    return chess.polyglot.zobrist_hash(board)


def print_best_line_san(best_line, board):
    san_best_line = []
    for move in best_line:
        san_best_line.append(board.san(move))
        board.push(move)
    print("Best line: ", san_best_line)


def is_quiet(board, move):
    return not board.is_capture(move) and not board.gives_check(move)


def is_draw(board):
    # Check if the move leads to a position that is already repeated
    if board.is_repetition():
        return True
    # Check for insufficient material
    if board.is_insufficient_material():
        return True
    # Check for stalemate
    if board.is_stalemate():
        return True
    # Check for 75-move rule
    if board.is_seventyfive_moves():
        return True
    return False


def is_claimable_draw(board):
    return (
        board.can_claim_threefold_repetition() or
        board.can_claim_fifty_moves()
    )


def is_endgame(board, piece_value):
    # Compute total non-pawn material for both sides
    material = 0
    for piece in board.piece_map().values():
        if piece.piece_type != chess.PAWN and piece.piece_type != chess.KING:
            material += piece_value[piece.piece_type]

    # Threshold: 2400 centipawns (~24 points = 2 rooks + minor)
    return material <= 2400
