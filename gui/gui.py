import pygame
import chess

# --- Config ---
WIDTH, HEIGHT = 640, 640
DIMENSION = 8
SQUARE_SIZE = WIDTH // DIMENSION
FPS = 15

# Colors
WHITE = (240, 217, 181)
BROWN = (181, 136, 99)

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


def is_promotion(board, move):
    piece = board.piece_at(move.from_square)
    if not piece or piece.piece_type != chess.PAWN:
        return False
    if piece.color == chess.WHITE and chess.square_rank(move.to_square) == 7:
        return True
    if piece.color == chess.BLACK and chess.square_rank(move.to_square) == 0:
        return True
    return False


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


def draw_game_over(screen, board):
    winner = "White" if board.result() == "1-0" else "Black" if board.result() == "0-1" else "Draw"

    # draw a cross over the defeated king
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece and piece.piece_type == chess.KING and piece.color == board.turn:
            col = chess.square_file(square)
            row = 7 - chess.square_rank(square)
            pygame.draw.line(screen, (255, 0, 0), (col * SQUARE_SIZE, row * SQUARE_SIZE),
                             ((col + 1) * SQUARE_SIZE, (row + 1) * SQUARE_SIZE), 5)
            pygame.draw.line(screen, (255, 0, 0), ((col + 1) * SQUARE_SIZE, row * SQUARE_SIZE),
                             (col * SQUARE_SIZE, (row + 1) * SQUARE_SIZE), 5)

    # display the winner in the middle of the board over pieces
    font = pygame.font.Font(None, 74)
    text = font.render(f"{winner} wins!", True, (255, 0, 0))
    text_rect = text.get_rect(center=(WIDTH // 2, HEIGHT // 2))
    screen.blit(text, text_rect)

    pygame.display.flip()


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

        # Highlight selected piece and legal moves
        if selected_square is not None:
            highlight_legal_moves(screen, board, selected_square)

        pygame.display.flip()
        clock.tick(FPS)

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
    main()
