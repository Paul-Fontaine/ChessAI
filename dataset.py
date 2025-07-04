import pickle
import torch
from torch.utils.data import Dataset


class PositionsDataset(Dataset):
    def __init__(self, file: str = "dataset.pkl"):
        data = pickle.load(open(file, "rb"))
        self.X_raw = data['X']
        self.y = data['y']

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        active_indices = self.X_raw[idx]
        x = torch.zeros((1, 832), dtype=torch.float32)
        x[0, list(active_indices)] = 1.0
        y = self.y[idx]
        y = torch.tensor(y, dtype=torch.float32).unsqueeze(0)
        return x, y


if __name__ == "__main__":
    import os
    import chess.pgn
    import chess.engine
    import torch
    from tqdm import tqdm

    from nnue import KPNNUE

    # Configuration
    PGN_FOLDER = "C:/Users/Utilisateur/Downloads/Lichess Elite Database/Lichess Elite Database"
    STOCKFISH_PATH = "C:/Program Files/stockfish/stockfish-windows-x86-64-avx2.exe"
    OUTPUT_PATH = "dataset.pkl"
    EVAL_EVERY_N_PLIES = 2
    MAX_SAMPLES = 100000

    # Initialize NNUE encoder
    nnue_encoder = KPNNUE()

    # Initialize Stockfish
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    engine.configure({"Threads": 4})

    positions = []
    evaluations = []
    count = 0

    for pgn_file in tqdm(os.listdir(PGN_FOLDER), desc="PGN files"):
        with open(os.path.join(PGN_FOLDER, pgn_file)) as f:
            while True:
                game = chess.pgn.read_game(f)
                if game is None:
                    break

                board = game.board()
                ply_count = 0
                for move in game.mainline_moves():
                    board.push(move)
                    ply_count += 1

                    if ply_count % EVAL_EVERY_N_PLIES != 0:
                        continue
                    if board.is_game_over():
                        continue

                    try:
                        # Evaluate using Stockfish without search
                        info = engine.analyse(board, chess.engine.Limit(depth=0))
                        score = info["score"].white()
                        eval_cp = score.score()
                    except Exception as e:
                        print(f"Engine error: {e}")
                        continue
                    if eval_cp is None:
                        continue

                    # Encode features
                    active_features = nnue_encoder.encode_kp(board)

                    positions.append(active_features)
                    evaluations.append(eval_cp)
                    count += 1

                    if count >= MAX_SAMPLES:
                        engine.quit()
                        with open(OUTPUT_PATH, "wb") as f:
                            pickle.dump({'X': positions, 'y': evaluations}, f)
                        exit(3)
