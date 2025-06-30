import pygame

from engine import ChessEngine
from gui.gui import *


def main(time_limit=4.0):
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Chess')
    clock = pygame.time.Clock()
    board = chess.Board()
    engine = ChessEngine()

    load_images('gui/pieces_images')
    selected_square = None

    running = True
    while running:
        draw_board(screen, board)

        # if a move was made, highlight the squares
        if board.move_stack:
            last_move = board.peek()
            piece_involved = board.piece_at(last_move.to_square)
            highlight_move_squares(screen, last_move, piece_involved)

        # Highlight selected piece and legal moves
        if selected_square is not None:
            highlight_legal_moves(screen, board, selected_square)

        pygame.display.flip()
        clock.tick(FPS)

        # Check if it's the engine's (Black's) turn
        if not board.is_game_over() and board.turn == chess.BLACK:
            engine_move = engine.search(board, time_limit=time_limit)
            if engine_move:
                board.push(engine_move)
                continue  # Skip player input on engine turn

        # Handle mouse clicks
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                pygame.quit()

            elif event.type == pygame.MOUSEBUTTONDOWN:
                square = get_square(pygame.mouse.get_pos())
                if selected_square is None:
                    # First click - select a piece
                    piece = board.piece_at(square)
                    if piece and (piece.color == board.turn):
                        selected_square = square
                else:
                    # Second click - try to make move
                    move = chess.Move(selected_square, square)
                    if is_promotion(board, move):
                        move.promotion = chess.QUEEN
                    if move in board.legal_moves:
                        board.push(move)
                    selected_square = None

        # check if there is a checkmate or a draw
        if board.is_checkmate():
            running = False
            winner = "White" if board.result() == "1-0" else "Black"
            end_message = f"Checkmate! \n{winner} wins"
        if board.is_stalemate():
            running = False
            end_message = "Stalemate! \nIt's a draw"
        if board.is_insufficient_material():
            running = False
            end_message = "Draw by \ninsufficient material"
        if board.is_seventyfive_moves():
            running = False
            end_message = "Draw by \n75-move rule"
        if board.is_fivefold_repetition():
            running = False
            end_message = "Draw by \nfivefold repetition"

    # end while

    # when the game is over, display the result
    # update the screen to show the final position
    draw_board(screen, board)
    pygame.display.flip()
    pygame.time.wait(200)
    draw_game_over(screen, board, end_message)
    pygame.display.flip()
    # Wait until the user closes the window
    endscreen = True
    while endscreen:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                endscreen = False

    pygame.quit()


if __name__ == "__main__":
    main(time_limit=50.0)
