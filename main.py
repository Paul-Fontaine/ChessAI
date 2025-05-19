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

        # Highlight selected piece and legal moves
        if selected_square is not None:
            highlight_legal_moves(screen, board, selected_square)

        pygame.display.flip()
        clock.tick(FPS)

        # Check if it's the engine's (Black's) turn
        if not board.is_game_over() and board.turn == chess.BLACK:
            engine_move, score, depth, node_count = engine.search(board, time_limit=time_limit)  # Engine gets 4 seconds
            if engine_move:
                board.push(engine_move)
                print(f"move: {engine_move} ; score: {score} ; depth reached: {depth} ; node_count: {node_count}")
            continue  # Skip player input on engine turn

        # Handle mouse clicks
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

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

        # check if the game is over
        if board.is_game_over():
            running = False
            # update the screen to show the final position
            draw_board(screen, board)
            pygame.display.flip()
            pygame.time.wait(200)
    # end while

    # when the game is over, display the result
    draw_game_over(screen, board)
    # Wait until the user closes the window
    endscreen = True
    while endscreen:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                endscreen = False

    pygame.quit()


if __name__ == "__main__":
    main(time_limit=10.0)
