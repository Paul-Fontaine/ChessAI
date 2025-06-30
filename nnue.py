import chess
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class KPNNUE(nn.Module):
    def __init__(self, input_dim=832, hidden1=256, hidden2=32):
        super().__init__()
        self.active_features_indices = set()
        self.accumulator = None  # Shared accumulator for feature transformation

        self.fc1 = nn.Linear(input_dim, hidden1)  # Feature transformation layer (shared accumulator)
        self.fc2 = nn.Linear(hidden1, hidden2)  # Second hidden layer
        self.fc_out = nn.Linear(hidden2, 1)  # Output layer

    def forward(self, x):
        # x: batch_size × input_dim
        x = F.relu(self.fc1(x))      # Feature transform
        x = F.relu(self.fc2(x))      # Additional abstraction
        x = self.fc_out(x)           # Final output
        return x  # Evaluation score in centipawns

    def forward_with_accumulator(self):
        x = F.relu(self.accumulator)
        x = F.relu(self.fc2(x))
        x = self.fc_out(x)
        return x

    @staticmethod
    def encode_kp(board: chess.Board) -> set:
        active = set()

        # 1) Encode king square of side-to-move (first 64 features)
        stm_king_sq = board.king(board.turn)
        active.add(stm_king_sq)

        # 2) Encode piece-square planes
        offset = 64
        for square, piece in board.piece_map().items():
            piece_type_offset = piece.piece_type-1 + (6 if piece.color == chess.BLACK else 0)
            idx = offset + piece_type_offset * 64 + square
            active.add(idx)

        return active  # list of int indices

    @staticmethod
    def tensor_from_active_indices(active_indices):
        features = torch.zeros((1, 832), dtype=torch.float32)
        features[0, list(active_indices)] = 1.0
        return features

    def init_accumulator(self, board):
        active_indices = self.encode_kp(board)
        self.active_features_indices = active_indices
        self.accumulator = self.fc1.weight[:, list(active_indices)].sum(dim=1) + self.fc1.bias

    def update_accumulator(self, board_after_move):
        features_indices_after = self.encode_kp(board_after_move)

        removed = self.active_features_indices - features_indices_after
        added = features_indices_after - self.active_features_indices

        for idx in added:
            self.accumulator += self.fc1.weight[:, idx]
        for idx in removed:
            self.accumulator -= self.fc1.weight[:, idx]

        self.active_features_indices = features_indices_after

    def get_evaluation(self, board):
        self.eval()
        active_features = self.encode_kp(board)
        features = torch.zeros((1, 832), dtype=torch.float32)
        features[0, list(active_features)] = 1.0
        with torch.no_grad():
            score = self.forward(features)
        return score.item()

    def get_evaluation_incremental(self):
        self.eval()
        with torch.no_grad():
            return self.forward_with_accumulator().item()

    def save(self, path="nnue_weights.pth"):
        torch.save(self.state_dict(), path)

    def load(self, path="nnue_weights.pth"):
        self.load_state_dict(torch.load(path, map_location=torch.device('cpu'), weights_only=True))
