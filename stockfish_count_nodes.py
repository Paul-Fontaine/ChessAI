import time
import matplotlib.pyplot as plt
import chess
import chess.engine

# Path to your Stockfish binary
STOCKFISH_PATH = "C:/Program Files/stockfish/stockfish-windows-x86-64-avx2.exe"


def run_stockfish_get_nodes(fen: str, max_depth: int = 12):
    board = chess.Board(fen)
    engine = chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH)
    nodes_per_depth = [(0, 0, 0, 0)] * max_depth
    cum_nodes = 0

    print(f"{'Depth':>5} | {'Nodes':>12} | {'Cumulative Nodes':>16} | {'Time (s)':>12} | {'Speed (nodes/s)':>12}")
    print("-" * 50)

    for depth in range(1, max_depth + 1):
        start_time = time.time()
        result = engine.analyse(board, chess.engine.Limit(depth=depth))
        elapsed_time = time.time() - start_time
        nodes = result.get("nodes", "N/A")
        cum_nodes += nodes
        speed = nodes / elapsed_time if elapsed_time > 0 else 0
        print(f"{depth:>5} | {nodes:>12} | {cum_nodes:>16} | {elapsed_time:>12.2f} | {speed:>12.0f}")
        nodes_per_depth[depth - 1] = (nodes, cum_nodes, elapsed_time, speed)

    engine.quit()
    # plot_result(nodes_per_depth)


def plot_result(nodes_per_depth):
    depths = list(range(1, len(nodes_per_depth) + 1))
    nodes = [x[0] for x in nodes_per_depth]
    cummulative_nodes = [x[1] for x in nodes_per_depth]
    times = [x[2] for x in nodes_per_depth]
    speeds = [x[3] for x in nodes_per_depth]

    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    plt.plot(depths, nodes, label='Nodes', marker='o')
    plt.title('Nodes per Depth')
    plt.xlabel('Depth')
    plt.ylabel('Nodes')
    plt.grid()

    plt.subplot(2, 2, 2)
    plt.plot(depths, cummulative_nodes, label='Cummulative Nodes', marker='o', color='orange')
    plt.title('Cummulative Nodes per Depth')
    plt.xlabel('Depth')
    plt.ylabel('Cummulative Nodes')
    plt.grid()

    plt.subplot(2, 2, 3)
    plt.plot(depths, times, label='Time (s)', marker='o', color='green')
    plt.title('Time per Depth')
    plt.xlabel('Depth')
    plt.ylabel('Time (s)')
    plt.grid()

    plt.subplot(2, 2, 4)
    plt.plot(depths, speeds, label='Speed (nodes/s)', marker='o', color='red')
    plt.title('Speed per Depth')
    plt.xlabel('Depth')
    plt.ylabel('Speed (nodes/s)')
    plt.grid()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    start_fen = chess.STARTING_FEN  # Standard starting position
    run_stockfish_get_nodes(start_fen, max_depth=20)
