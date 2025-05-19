# ChessAI

## Description
I made a chess engine similar to Stockfish, it uses a minimax tree search 
with alpha-beta pruning and other search optimizations. 
The evaluation function is a simple heuristic that counts material.

## requirements
- python 3.8+
- [python-chess](https://pypi.org/project/chess/) for the moves management
- [pygame](https://pypi.org/project/pygame/) for the GUI  
`pip install python-chess pygame`

## Usage
Run the `main.py` file, it will open a window with the chess board. User play whites pieces and the engine play blacks.
You can set the maximum time allowed to the engine to play by editing `main.py`