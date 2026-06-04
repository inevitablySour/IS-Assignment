#!/usr/bin/env python3
"""
Analyze game logs from the Alpha-Beta agent.
Reads game_logs.json and produces:
  - Console summary of all key metrics
  - Plots saved to analysis_plots/ directory
"""

import json
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from datetime import datetime

# ── Configuration ─────────────────────────────────────────────────────
PLOT_DIR = "analysis_plots"

# ── Load data (from SQLite) ─────────────────────────────────────────

def load_games(game_type: str = None) -> list:
    """Load games from SQLite DB, enriched with agent_stats per game."""
    from game_db import GameDB
    db = GameDB()
    games = db.get_all_games(game_type=game_type)
    for g in games:
        g["agent_search_stats"] = db.get_agent_stats(g["game_id"])
    db.close()
    return games


# ── Metric extraction ───────────────────────────────────────────────

def extract_metrics(games: list) -> dict:
    """Compute all relevant metrics from the game log."""
    m = {}

    # --- Overall record ---
    results = [g["result"] for g in games]
    m["total_games"] = len(games)
    m["wins"] = results.count("win")
    m["losses"] = results.count("loss")
    m["draws"] = results.count("draw")
    m["timeouts"] = results.count("timeout")
    m["win_rate"] = m["wins"] / m["total_games"] * 100 if m["total_games"] else 0

    # --- By symbol ---
    by_sym = {}
    for sym in ("X", "O"):
        sg = [g for g in games if g.get("my_symbol") == sym]
        w = sum(1 for g in sg if g["result"] == "win")
        by_sym[sym] = {"games": len(sg), "wins": w,
                       "win_rate": w / len(sg) * 100 if sg else 0}
    m["by_symbol"] = by_sym

    # --- Game length ---
    lengths = [g.get("total_moves", 0) for g in games]
    m["game_lengths"] = lengths
    m["avg_game_length"] = np.mean(lengths)
    m["median_game_length"] = np.median(lengths)
    m["min_game_length"] = min(lengths)
    m["max_game_length"] = max(lengths)
    m["std_game_length"] = np.std(lengths)

    # --- Search stats (per-move) ---
    all_depths = []
    all_nodes = []
    all_times = []
    per_game_avg_depth = []
    per_game_avg_nodes = []
    per_game_avg_time = []

    for g in games:
        stats = g.get("agent_search_stats", [])
        depths = [s["depth_reached"] for s in stats]
        nodes = [s["nodes_searched"] for s in stats]
        times = [s["time_remaining"] for s in stats]

        all_depths.extend(depths)
        all_nodes.extend(nodes)
        all_times.extend(times)

        if depths:
            per_game_avg_depth.append(np.mean(depths))
            per_game_avg_nodes.append(np.mean(nodes))
        if len(times) >= 2:
            # Compute average time spent per move (delta between consecutive remaining times)
            time_spent = [times[i] - times[i + 1] for i in range(len(times) - 1)]
            per_game_avg_time.append(np.mean(time_spent) if time_spent else 0)

    m["all_depths"] = all_depths
    m["all_nodes"] = all_nodes
    m["avg_depth"] = np.mean(all_depths) if all_depths else 0
    m["max_depth"] = max(all_depths) if all_depths else 0
    m["avg_nodes_per_move"] = np.mean(all_nodes) if all_nodes else 0
    m["max_nodes"] = max(all_nodes) if all_nodes else 0
    m["per_game_avg_depth"] = per_game_avg_depth
    m["per_game_avg_nodes"] = per_game_avg_nodes
    m["per_game_avg_time_per_move"] = per_game_avg_time

    # --- Forced moves (depth 0 = instant win/block) ---
    forced = sum(1 for d in all_depths if d == 0)
    m["forced_moves"] = forced
    m["forced_move_pct"] = forced / len(all_depths) * 100 if all_depths else 0

    # --- Move types (first move as X) ---
    first_as_x = sum(1 for g in games if g.get("my_symbol") == "X")
    m["games_as_first_player"] = first_as_x
    m["games_as_second_player"] = m["total_games"] - first_as_x

    # --- Results list for per-game plots ---
    m["results"] = results
    m["game_ids_short"] = [g["game_id"][:8] for g in games]

    return m


# ── Console report ──────────────────────────────────────────────────

def print_report(m: dict):
    print("=" * 64)
    print("  GAME ANALYSIS REPORT — Alpha-Beta Agent")
    print("=" * 64)

    print(f"\n{'─'*40}")
    print("  OVERALL RECORD")
    print(f"{'─'*40}")
    print(f"  Games played:    {m['total_games']}")
    print(f"  Wins:            {m['wins']}")
    print(f"  Losses:          {m['losses']}")
    print(f"  Draws:           {m['draws']}")
    print(f"  Timeouts:        {m['timeouts']}")
    print(f"  Win rate:        {m['win_rate']:.1f}%")

    print(f"\n{'─'*40}")
    print("  RECORD BY SYMBOL")
    print(f"{'─'*40}")
    for sym, info in m["by_symbol"].items():
        print(f"  As {sym}:  {info['wins']}W / {info['games']} games  "
              f"({info['win_rate']:.1f}%)")
    print(f"  First player (X): {m['games_as_first_player']} games")
    print(f"  Second player (O): {m['games_as_second_player']} games")

    print(f"\n{'─'*40}")
    print("  GAME LENGTH")
    print(f"{'─'*40}")
    print(f"  Average:         {m['avg_game_length']:.1f} moves")
    print(f"  Median:          {m['median_game_length']:.0f} moves")
    print(f"  Shortest:        {m['min_game_length']} moves")
    print(f"  Longest:         {m['max_game_length']} moves")
    print(f"  Std deviation:   {m['std_game_length']:.1f} moves")

    print(f"\n{'─'*40}")
    print("  SEARCH PERFORMANCE")
    print(f"{'─'*40}")
    print(f"  Avg depth:       {m['avg_depth']:.1f}")
    print(f"  Max depth:       {m['max_depth']}")
    print(f"  Avg nodes/move:  {m['avg_nodes_per_move']:,.0f}")
    print(f"  Max nodes:       {m['max_nodes']:,}")
    print(f"  Forced moves:    {m['forced_moves']}  "
          f"({m['forced_move_pct']:.1f}% of all moves)")

    if m['per_game_avg_time_per_move']:
        avg_time = np.mean(m['per_game_avg_time_per_move'])
        print(f"  Avg time/move:   {avg_time:.1f}s")

    print(f"\n{'─'*40}")
    print("  PER-GAME BREAKDOWN")
    print(f"{'─'*40}")
    print(f"  {'#':<4} {'Result':<8} {'Moves':<7} {'Avg Depth':<11} {'Avg Nodes':<12}")
    print(f"  {'─'*44}")
    for i, (r, l, d, n) in enumerate(zip(
        m["results"], m["game_lengths"],
        m["per_game_avg_depth"], m["per_game_avg_nodes"]
    ), 1):
        print(f"  {i:<4} {r:<8} {l:<7} {d:<11.1f} {n:<12,.0f}")

    print()


# ── Plotting ────────────────────────────────────────────────────────

def generate_plots(m: dict, games: list, plot_dir: str = PLOT_DIR):
    os.makedirs(plot_dir, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")
    game_nums = list(range(1, m["total_games"] + 1))

    # 1. Win/Loss/Draw pie chart
    fig, ax = plt.subplots(figsize=(6, 6))
    labels, sizes, colors = [], [], []
    for label, key, color in [("Wins", "wins", "#4CAF50"),
                               ("Losses", "losses", "#F44336"),
                               ("Draws", "draws", "#FFC107"),
                               ("Timeouts", "timeouts", "#9E9E9E")]:
        if m[key] > 0:
            labels.append(f"{label} ({m[key]})")
            sizes.append(m[key])
            colors.append(color)
    ax.pie(sizes, labels=labels, colors=colors, autopct="%1.0f%%",
           startangle=90, textprops={"fontsize": 12})
    ax.set_title("Game Outcomes", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{plot_dir}/01_outcomes_pie.png", dpi=150)
    plt.close(fig)

    # 2. Game length bar chart
    fig, ax = plt.subplots(figsize=(10, 5))
    result_colors = {"win": "#4CAF50", "loss": "#F44336",
                     "draw": "#FFC107", "timeout": "#9E9E9E"}
    bar_colors = [result_colors.get(r, "#999") for r in m["results"]]
    ax.bar(game_nums, m["game_lengths"], color=bar_colors, edgecolor="white")
    ax.axhline(m["avg_game_length"], color="#333", linestyle="--",
               linewidth=1, label=f"Average ({m['avg_game_length']:.1f})")
    ax.set_xlabel("Game Number", fontsize=12)
    ax.set_ylabel("Total Moves", fontsize=12)
    ax.set_title("Game Length per Game", fontsize=14, fontweight="bold")
    ax.set_xticks(game_nums)
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{plot_dir}/02_game_lengths.png", dpi=150)
    plt.close(fig)

    # 3. Search depth over time (per-game average)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(game_nums[:len(m["per_game_avg_depth"])],
            m["per_game_avg_depth"], "o-", color="#1976D2",
            markersize=6, linewidth=1.5)
    ax.set_xlabel("Game Number", fontsize=12)
    ax.set_ylabel("Average Search Depth", fontsize=12)
    ax.set_title("Average Search Depth per Game", fontsize=14, fontweight="bold")
    ax.set_xticks(game_nums)
    fig.tight_layout()
    fig.savefig(f"{plot_dir}/03_avg_depth_per_game.png", dpi=150)
    plt.close(fig)

    # 4. Nodes searched per game (average)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(game_nums[:len(m["per_game_avg_nodes"])],
           m["per_game_avg_nodes"], color="#FF9800", edgecolor="white")
    ax.set_xlabel("Game Number", fontsize=12)
    ax.set_ylabel("Average Nodes per Move", fontsize=12)
    ax.set_title("Average Nodes Searched per Move (by Game)",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(game_nums)
    fig.tight_layout()
    fig.savefig(f"{plot_dir}/04_avg_nodes_per_game.png", dpi=150)
    plt.close(fig)

    # 5. Depth distribution histogram
    fig, ax = plt.subplots(figsize=(8, 5))
    depth_counts = Counter(m["all_depths"])
    depths_sorted = sorted(depth_counts.keys())
    ax.bar(depths_sorted, [depth_counts[d] for d in depths_sorted],
           color="#7B1FA2", edgecolor="white")
    ax.set_xlabel("Search Depth Reached", fontsize=12)
    ax.set_ylabel("Number of Moves", fontsize=12)
    ax.set_title("Distribution of Search Depth Across All Moves",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(depths_sorted)
    fig.tight_layout()
    fig.savefig(f"{plot_dir}/05_depth_distribution.png", dpi=150)
    plt.close(fig)

    # 6. Nodes searched distribution (log scale)
    fig, ax = plt.subplots(figsize=(8, 5))
    nodes_nonzero = [n for n in m["all_nodes"] if n > 0]
    if nodes_nonzero:
        ax.hist(nodes_nonzero, bins=30, color="#00897B", edgecolor="white")
        ax.set_xlabel("Nodes Searched", fontsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.set_title("Distribution of Nodes Searched per Move",
                     fontsize=14, fontweight="bold")
        ax.set_xscale("log")
    fig.tight_layout()
    fig.savefig(f"{plot_dir}/06_nodes_distribution.png", dpi=150)
    plt.close(fig)

    # 7. Depth progression within a single game (longest game)
    longest_idx = np.argmax(m["game_lengths"])
    longest = games[longest_idx]
    stats = longest.get("agent_search_stats", [])
    if stats:
        fig, ax1 = plt.subplots(figsize=(10, 5))
        moves = [s["move_number"] for s in stats]
        depths = [s["depth_reached"] for s in stats]
        nodes = [s["nodes_searched"] for s in stats]

        color1 = "#1976D2"
        ax1.set_xlabel("Move Number", fontsize=12)
        ax1.set_ylabel("Search Depth", fontsize=12, color=color1)
        ax1.plot(moves, depths, "o-", color=color1, label="Depth", markersize=5)
        ax1.tick_params(axis="y", labelcolor=color1)

        ax2 = ax1.twinx()
        color2 = "#FF9800"
        ax2.set_ylabel("Nodes Searched", fontsize=12, color=color2)
        ax2.bar(moves, nodes, alpha=0.3, color=color2, label="Nodes", width=1.5)
        ax2.tick_params(axis="y", labelcolor=color2)

        ax1.set_title(f"Search Progression — Longest Game "
                      f"({longest.get('total_moves', '?')} moves)",
                      fontsize=14, fontweight="bold")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
        fig.tight_layout()
        fig.savefig(f"{plot_dir}/07_longest_game_progression.png", dpi=150)
        plt.close(fig)

    # 8. Game length by symbol (box plot)
    x_lengths = [g["total_moves"] for g in games if g.get("my_symbol") == "X"]
    o_lengths = [g["total_moves"] for g in games if g.get("my_symbol") == "O"]
    fig, ax = plt.subplots(figsize=(7, 5))
    bp = ax.boxplot([x_lengths, o_lengths], tick_labels=["Playing as X", "Playing as O"],
                    patch_artist=True, widths=0.5)
    bp["boxes"][0].set_facecolor("#2196F3")
    bp["boxes"][1].set_facecolor("#F44336")
    for box in bp["boxes"]:
        box.set_alpha(0.6)
    ax.set_ylabel("Total Moves", fontsize=12)
    ax.set_title("Game Length by Symbol", fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(f"{plot_dir}/08_length_by_symbol.png", dpi=150)
    plt.close(fig)

    # 9. Cumulative win rate over time
    fig, ax = plt.subplots(figsize=(10, 5))
    cumulative_wins = np.cumsum([1 if r == "win" else 0 for r in m["results"]])
    cumulative_wr = cumulative_wins / np.arange(1, m["total_games"] + 1) * 100
    ax.plot(game_nums, cumulative_wr, "o-", color="#4CAF50",
            markersize=5, linewidth=2)
    ax.axhline(100, color="#ccc", linestyle=":", linewidth=1)
    ax.set_xlabel("Games Played", fontsize=12)
    ax.set_ylabel("Cumulative Win Rate (%)", fontsize=12)
    ax.set_title("Win Rate Over Time", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.set_xticks(game_nums)
    fig.tight_layout()
    fig.savefig(f"{plot_dir}/09_cumulative_win_rate.png", dpi=150)
    plt.close(fig)


    print(f"  9 plots saved to {plot_dir}/")


# ── Main ────────────────────────────────────────────────────────────

def main():
    if not os.path.exists("game_logs.db"):
        print("Error: game_logs.db not found. Play some games first.")
        print("  (Run 'python game_db.py' to migrate from game_logs.json)")
        sys.exit(1)

    games = load_games()
    if not games:
        print("No games in the log file.")
        sys.exit(1)

    metrics = extract_metrics(games)
    print_report(metrics)
    generate_plots(metrics, games)


if __name__ == "__main__":
    main()
