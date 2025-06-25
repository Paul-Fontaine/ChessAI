import pygame
import chess
from board_helper_functions import is_promotion

# --- Config ---
WIDTH, HEIGHT = 640, 640
DIMENSION = 8
SQUARE_SIZE = WIDTH // DIMENSION
FPS = 15

# Colors
WHITE = (240, 217, 181)
BROWN = (181, 136, 99)
LIGHT_YELLOW = (240, 240, 50)

# Load piece images
IMAGES = {}


def load_images(pieces_images_folder_path='pieces_images/'):
    pieces = ['r', 'n', 'b', 'q', 'k', 'p', 'R', 'N', 'B', 'Q', 'K', 'P']
    for piece in pieces:
        if not pieces_images_folder_path.endswith('/'):
            pieces_images_folder_path += '/'
        color = 'w' if piece.isupper() else 'b'
        name = piece.upper()
        filename = f"{pieces_images_folder_path}{color}{name}.png"
        IMAGES[piece] = pygame.transform.scale(
            pygame.image.load(filename), (SQUARE_SIZE, SQUARE_SIZE)
        )


# Draw the board
def draw_board(screen, board):
    colors = [WHITE, BROWN]
    for r in range(DIMENSION):
        for c in range(DIMENSION):
            color = colors[(r + c) % 2]
            pygame.draw.rect(screen, color, pygame.Rect(c * SQUARE_SIZE, r * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))

    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            piece_str = piece.symbol()
            col = chess.square_file(square)
            row = 7 - chess.square_rank(square)
            screen.blit(IMAGES[piece_str], pygame.Rect(col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))


# Convert mouse position to square index
def get_square(pos):
    x, y = pos
    col = x // SQUARE_SIZE
    row = y // SQUARE_SIZE
    return chess.square(col, 7 - row)


def highlight_legal_moves(screen, board, selected_square):
    # Highlight the selected square
    col = chess.square_file(selected_square)
    row = 7 - chess.square_rank(selected_square)
    pygame.draw.rect(screen, (0, 255, 0), pygame.Rect(
        col * SQUARE_SIZE, row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE), 3)

    # Highlight legal moves with a dots
    for move in board.legal_moves:
        if move.from_square == selected_square:
            to_col = chess.square_file(move.to_square)
            to_row = 7 - chess.square_rank(move.to_square)
            center_x = to_col * SQUARE_SIZE + SQUARE_SIZE // 2
            center_y = to_row * SQUARE_SIZE + SQUARE_SIZE // 2

            # If it's a capture, draw a red circle around the piece
            if board.piece_at(move.to_square):
                pygame.draw.circle(screen, (255, 0, 0), (center_x, center_y), SQUARE_SIZE // 2, 3)

            # Draw a dot in the center of the square
            else:
                pygame.draw.circle(screen, (0, 0, 0), (center_x, center_y), SQUARE_SIZE // 8)


def highlight_move_squares(screen, move, piece):
    # Highlight the starting square of the move
    from_col = chess.square_file(move.from_square)
    from_row = 7 - chess.square_rank(move.from_square)
    highlight_surface_from = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
    highlight_surface_from.fill((*LIGHT_YELLOW, 200))
    screen.blit(highlight_surface_from, (from_col * SQUARE_SIZE, from_row * SQUARE_SIZE))

    # Highlight the square where the piece will move
    to_col = chess.square_file(move.to_square)
    to_row = 7 - chess.square_rank(move.to_square)
    highlight_surface_to = pygame.Surface((SQUARE_SIZE, SQUARE_SIZE), pygame.SRCALPHA)
    highlight_surface_to.fill((*LIGHT_YELLOW, 100))
    screen.blit(highlight_surface_to, (to_col * SQUARE_SIZE, to_row * SQUARE_SIZE))

    # redraw the piece on the destination square above the highlight
    screen.blit(IMAGES[piece.symbol()], pygame.Rect(to_col * SQUARE_SIZE, to_row * SQUARE_SIZE, SQUARE_SIZE, SQUARE_SIZE))


def draw_game_over(screen, board, message):
    # draw a cross over the defeated king if there is a checkmate
    if board.is_checkmate():
        king_square = board.king(board.turn)
        col = chess.square_file(king_square)
        row = 7 - chess.square_rank(king_square)
        pygame.draw.line(screen, (255, 0, 0), (col * SQUARE_SIZE, row * SQUARE_SIZE),
                         ((col + 1) * SQUARE_SIZE, (row + 1) * SQUARE_SIZE), 5)
        pygame.draw.line(screen, (255, 0, 0), ((col + 1) * SQUARE_SIZE, row * SQUARE_SIZE),
                         (col * SQUARE_SIZE, (row + 1) * SQUARE_SIZE), 5)

    # display the winner or draw in the middle of the board over pieces
    font = pygame.font.Font(None, 74)
    lines = message.split('\n')
    line_height = font.get_linesize()
    for i, line in enumerate(lines):
        text = font.render(line, True, (255, 0, 0))
        text_rect = text.get_rect(center=(WIDTH // 2, HEIGHT // 2 + (i - len(lines) // 2) * line_height))
        screen.blit(text, text_rect)


# --- Main Program ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Chess')
    clock = pygame.time.Clock()
    board = chess.Board()

    load_images()
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
    main()
