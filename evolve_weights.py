#!/usr/bin/env python3
"""
Hill-climbing weight tuner for the Alpha-Beta agent.

Each round generates N mutants (one per physical CPU core) and evaluates all
of them against the current best simultaneously in one pool.map call.
The mutant with the most wins is adopted if it beats current.
This keeps all cores busy throughout — no idle gaps between rounds.

Usage:
    python evolve_weights.py                      # 100 rounds, 4 games/mutant
    python evolve_weights.py --rounds 50          # fewer rounds
    python evolve_weights.py --games 6 --time 2   # more games per mutant, faster
    python evolve_weights.py --fresh              # restart from default weights
"""

import argparse
import copy
import json
import multiprocessing as mp
import os
import random
import sys
import time
import uuid
import importlib
import importlib.util

# ── Game engine ──────────────────────────────────────────────────────

WIN = 6
DIRECTIONS = [(0, 1), (1, 0), (1, 1), (1, -1)]


def check_win(grid, row, col, symbol):
    for dr, dc in DIRECTIONS:
        count = 1
        for i in range(1, WIN):
            if grid.get((row + dr * i, col + dc * i)) == symbol:
                count += 1
            else:
                break
        for i in range(1, WIN):
            if grid.get((row - dr * i, col - dc * i)) == symbol:
                count += 1
            else:
                break
        if count >= WIN:
            return True
    return False


def make_state(grid, symbol, opp, move_count, time_limit):
    str_grid = {f"({r}, {c})": v for (r, c), v in grid.items()}
    return {
        "grid": str_grid,
        "your_symbol": symbol,
        "opponent_symbol": opp,
        "your_turn": True,
        "move_count": move_count,
        "time_remaining": time_limit,
        "game_status": "active",
    }


def play_game(agent_x, agent_o, time_limit=1.0, max_moves=200):
    """Play one game silently. Returns 'X', 'O', or 'DRAW'."""
    grid = {}
    agents = {"X": agent_x, "O": agent_o}
    turn = "X"
    move_count = 0

    while move_count < max_moves:
        opp = "O" if turn == "X" else "X"
        state = make_state(grid, turn, opp, move_count, time_limit)
        row, col = agents[turn].get_move(state)

        if (row, col) in grid:
            return opp  # forfeit on illegal move

        grid[(row, col)] = turn
        move_count += 1

        if check_win(grid, row, col, turn):
            return turn

        turn = opp

    return "DRAW"


# ── Agent loading with weight injection ──────────────────────────────

def load_module():
    base = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base, "student_agent.py")
    spec = importlib.util.spec_from_file_location("student_agent_evol", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_agent(mod, weights, time_cap=1.0):
    mod.ATTACK = list(weights["ATTACK"])
    mod.DEFEND = list(weights["DEFEND"])
    mod.CONVERGE = list(weights["CONVERGE"])
    mod.FORK_BONUS = weights["FORK_BONUS"]
    mod.TIME_CAP = time_cap
    return mod.StudentAgent()


# ── Weight operations ─────────────────────────────────────────────────

DEFAULT_WEIGHTS = {
    "ATTACK":     [0, 1, 12, 120, 1_200, 120_000, 10_000_000],
    "DEFEND":     [0, 2, 15, 150, 1_500, 150_000, 10_000_000],
    "CONVERGE":   [0, 0, 5_000, 20_000, 80_000],
    "FORK_BONUS": 500_000,
}


def mutate(weights, strength=0.3):
    w = copy.deepcopy(weights)
    target = random.choice(["ATTACK", "DEFEND", "CONVERGE", "FORK_BONUS"])

    if target == "FORK_BONUS":
        factor = random.uniform(1 - strength, 1 + strength)
        w["FORK_BONUS"] = max(1000, int(w["FORK_BONUS"] * factor))
    else:
        arr = w[target]
        mutable = list(range(1, len(arr) - 1))
        n_mutate = random.randint(1, min(2, len(mutable)))
        for idx in random.sample(mutable, n_mutate):
            factor = random.uniform(1 - strength, 1 + strength)
            arr[idx] = max(1, int(arr[idx] * factor))
        for i in range(2, len(arr) - 1):
            if arr[i] < arr[i - 1]:
                arr[i] = arr[i - 1] + 1

    return w


# ── Worker ────────────────────────────────────────────────────────────

def _play_one_game(args_tuple):
    """
    Worker: plays one game between weights_a and weights_b.
    Returns 'a', 'b', or 'draw'.
    """
    weights_a, weights_b, time_cap, a_plays_x = args_tuple
    import io
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()

    mod_a = load_module()
    mod_b = load_module()
    agent_a = make_agent(mod_a, weights_a, time_cap)
    agent_b = make_agent(mod_b, weights_b, time_cap)

    if a_plays_x:
        winner = play_game(agent_a, agent_b, time_cap)
        if winner == "X":   return "a"
        elif winner == "O": return "b"
        else:               return "draw"
    else:
        winner = play_game(agent_b, agent_a, time_cap)
        if winner == "X":   return "b"
        elif winner == "O": return "a"
        else:               return "draw"


# ── Match (all mutants in one pool.map call) ──────────────────────────

def play_match(current, mutants, games_per_mutant, time_cap, pool):
    """
    Evaluate every mutant against current simultaneously.
    Submits len(mutants) * games_per_mutant tasks in one pool.map call so
    workers always have queued work and the tail-idle problem is minimised.

    Returns list of (mutant_wins, current_wins, draws) per mutant.
    'a' = current won, 'b' = mutant won (current is always weights_a).
    """
    tasks = [
        (current, mutant, time_cap, gi % 2 == 0)
        for mutant in mutants
        for gi in range(games_per_mutant)
    ]

    results = pool.map(_play_one_game, tasks)

    scores = []
    for mi in range(len(mutants)):
        chunk = results[mi * games_per_mutant:(mi + 1) * games_per_mutant]
        scores.append((chunk.count("b"), chunk.count("a"), chunk.count("draw")))
    return scores


# ── Persistence ───────────────────────────────────────────────────────

WEIGHTS_FILE = "evolved_weights.json"


def save_weights(weights, round_num, record):
    data = {
        "weights": weights,
        "round": round_num,
        "record": record,
        "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(WEIGHTS_FILE, "w") as f:
        json.dump(data, f, indent=2)


def load_weights():
    if os.path.exists(WEIGHTS_FILE):
        try:
            with open(WEIGHTS_FILE) as f:
                data = json.load(f)
            print(f"  Loaded weights from {WEIGHTS_FILE} (round {data.get('round', '?')})")
            return data["weights"]
        except Exception:
            pass
    return copy.deepcopy(DEFAULT_WEIGHTS)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Hill-climbing weight tuner")
    parser.add_argument("--rounds", type=int, default=100,
                        help="Number of mutation rounds (default 100)")
    parser.add_argument("--games", type=int, default=4,
                        help="Games per mutant per round (default 4, must be even)")
    parser.add_argument("--time", type=float, default=1.0,
                        help="Time cap per move in seconds (default 1)")
    parser.add_argument("--strength", type=float, default=0.3,
                        help="Mutation strength 0-1 (default 0.3)")
    parser.add_argument("--fresh", action="store_true",
                        help="Start from default weights instead of loading saved")
    args = parser.parse_args()

    if args.games % 2 != 0:
        args.games += 1

    # One process per physical core; SMT doesn't help CPU-bound search
    n_workers = mp.cpu_count() // 2
    # One mutant per worker → total tasks = n_workers * games_per_mutant
    n_mutants = n_workers
    games_per_mutant = args.games

    load_module()  # verify student_agent loads cleanly before spawning pool

    print("=" * 60)
    print("  Hill-Climbing Weight Tuner")
    print("=" * 60)
    print(f"  Rounds:       {args.rounds}")
    print(f"  Mutants/round:{n_mutants}  (one per physical core)")
    print(f"  Games/mutant: {games_per_mutant}")
    print(f"  Tasks/round:  {n_mutants * games_per_mutant}  across {n_workers} workers")
    print(f"  Time/move:    {args.time}s  |  Mutation strength: {args.strength}")
    print("=" * 60)

    current = load_weights() if not args.fresh else copy.deepcopy(DEFAULT_WEIGHTS)
    improvements = 0
    total_games = 0

    from game_db import GameDB
    db = GameDB()

    print("\n  Starting weights:")
    for k, v in current.items():
        print(f"    {k}: {v}")
    print()

    with mp.Pool(processes=n_workers) as pool:
        for rnd in range(1, args.rounds + 1):
            mutants = [mutate(current, args.strength) for _ in range(n_mutants)]

            t0 = time.time()
            scores = play_match(current, mutants, games_per_mutant, args.time, pool)
            elapsed = time.time() - t0
            total_games += n_mutants * games_per_mutant

            # Best mutant by win count against current
            best_mi, (best_mw, best_cw, best_d) = max(
                enumerate(scores), key=lambda x: x[1][0]
            )

            game_id = f"evolve-r{rnd}-{uuid.uuid4().hex[:8]}"
            db.insert_game(
                game_id=game_id,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                game_type="evolve",
                my_symbol="*", opponent_symbol="*",
                result=f"best_mutant={best_mw} current={best_cw} draw={best_d}",
                total_moves=0,
                agent_name="current",
                opponent_name="mutants",
                config_notes=json.dumps({
                    "n_mutants": n_mutants,
                    "round": rnd,
                    "best_mutant_idx": best_mi,
                }),
            )
            db.commit()

            if best_mw > best_cw:
                best_mutant = mutants[best_mi]
                diffs = [
                    f"{k}: {current[k]} -> {best_mutant[k]}"
                    for k in current if current[k] != best_mutant[k]
                ]
                current = best_mutant
                improvements += 1
                save_weights(current, rnd, {
                    "improvements": improvements,
                    "total_rounds": rnd,
                    "total_games": total_games,
                })
                print(f"  R{rnd:>3}: ★ IMPROVED  "
                      f"best={best_mw}-{best_cw}-{best_d}  "
                      f"({elapsed:.0f}s)  {'; '.join(diffs)}")
            else:
                print(f"  R{rnd:>3}:   no gain    "
                      f"best={best_mw}-{best_cw}-{best_d}  "
                      f"({elapsed:.0f}s)")

    db.close()

    print(f"\n{'='*60}")
    print(f"  DONE — {args.rounds} rounds, {total_games} games")
    print(f"  Improvements: {improvements}")
    print(f"{'='*60}")
    print("  Final weights:")
    for k, v in current.items():
        print(f"    {k}: {v}")
    print(f"\n  Saved to {WEIGHTS_FILE}")
    save_weights(current, args.rounds, {
        "improvements": improvements,
        "total_rounds": args.rounds,
        "total_games": total_games,
    })


if __name__ == "__main__":
    main()
