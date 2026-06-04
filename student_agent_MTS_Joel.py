"""
Student Agent for Infinite Tic-Tac-Toe

Algorithm: MCTS with PUCT selection, biased rollouts, and tree reuse
- Dynamic Bounding Box: candidate moves restricted to active play area
- Sliding Window Threat Evaluator: O(cells*4) pattern scoring for rollouts
- PUCT selection: threat-prior-weighted exploration (AlphaZero-style)
- Forced move pre-check: instant win/block before entering MCTS
- Tree reuse: re-roots the MCTS tree on the opponent's reply each turn
"""

from client import GameClient
from typing import Tuple, Optional, List
import json
import math
import os
import random
import sys
import time

WIN = 6
DIRECTIONS = [(0, 1), (1, 0), (1, 1), (1, -1)]

C_PUCT = 1.5        # PUCT exploration constant
MAX_ROLLOUT_DEPTH = 50  # deep enough to reach terminals regularly
BB_MARGIN = 3       # cells outside occupied area to include in bbox
MAX_CANDIDATES = 20 # cap on moves expanded per node
ADVERSARIAL_PLIES = 4   # first N plies of each rollout use adversarial sampling
SOFTMAX_TEMP = 200.0    # softmax temperature for adversarial phase (higher = more uniform)


# ---------------------------------------------------------------------------
# Win detection
# ---------------------------------------------------------------------------

def is_winning_move(board: dict, move: Tuple[int, int], player: str) -> bool:
    r, c = move
    for dr, dc in DIRECTIONS:
        count = 1
        for i in range(1, WIN):
            if board.get((r + dr * i, c + dc * i)) == player:
                count += 1
            else:
                break
        for i in range(1, WIN):
            if board.get((r - dr * i, c - dc * i)) == player:
                count += 1
            else:
                break
        if count >= WIN:
            return True
    return False


def find_forced_move(board: dict, candidates: list,
                     my: str, opp: str) -> Optional[Tuple[int, int]]:
    for move in candidates:
        if is_winning_move(board, move, my):
            return move
    for move in candidates:
        if is_winning_move(board, move, opp):
            return move
    return None


# ---------------------------------------------------------------------------
# Dynamic Bounding Box
# ---------------------------------------------------------------------------

def compute_bbox(board: dict, margin: int = BB_MARGIN) -> Tuple[int, int, int, int]:
    if not board:
        return -margin, margin, -margin, margin
    rows = [r for r, _ in board]
    cols = [c for _, c in board]
    return min(rows) - margin, max(rows) + margin, min(cols) - margin, max(cols) + margin


def get_candidates(board: dict, bbox: Tuple[int, int, int, int]) -> List[Tuple[int, int]]:
    """Empty cells inside bbox that are adjacent to an occupied cell."""
    min_r, max_r, min_c, max_c = bbox
    occupied = set(board.keys())
    seen = set()
    result = []
    for r, c in occupied:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                pos = (r + dr, c + dc)
                if pos in seen or pos in occupied:
                    continue
                if min_r <= pos[0] <= max_r and min_c <= pos[1] <= max_c:
                    seen.add(pos)
                    result.append(pos)
    return result


# ---------------------------------------------------------------------------
# Threat prior (PUCT prior + move ordering)
# ---------------------------------------------------------------------------

def threat_prior(board: dict, move: Tuple[int, int], player: str, opponent: str) -> float:
    r, c = move
    score = 0.0
    for dr, dc in DIRECTIONS:
        my_count = 0
        opp_count = 0
        for i in range(1, WIN):
            sym = board.get((r + dr * i, c + dc * i))
            if sym == player:
                my_count += 1
            elif sym is not None:
                opp_count += 1
                break
            else:
                break
        for i in range(1, WIN):
            sym = board.get((r - dr * i, c - dc * i))
            if sym == player:
                my_count += 1
            elif sym is not None:
                opp_count += 1
                break
            else:
                break
        score += my_count ** 2 * 2.0 + opp_count ** 2 * 3.0
    return score + 1.0


def compute_priors(board: dict, candidates: list, player: str, opponent: str) -> dict:
    raw = {m: threat_prior(board, m, player, opponent) for m in candidates}
    total = sum(raw.values())
    return {m: v / total for m, v in raw.items()}


# ---------------------------------------------------------------------------
# Fast inline terminal check for rollouts (avoids full board scan)
# ---------------------------------------------------------------------------

def fast_static_score(board: dict, move: Tuple[int, int], my: str, opp: str) -> float:
    """
    After a rollout ends without a winner, score by counting the longest
    unblocked run for each player near the last move played.
    Returns a value in [-0.3, 0.3] from my perspective.
    """
    r, c = move
    my_best = 0
    opp_best = 0
    for dr, dc in DIRECTIONS:
        for player in (my, opp):
            count = 0
            for i in range(-(WIN - 1), WIN):
                sym = board.get((r + dr * i, c + dc * i))
                if sym == player:
                    count += 1
                else:
                    count = 0
            if player == my:
                my_best = max(my_best, count)
            else:
                opp_best = max(opp_best, count)
    if my_best > opp_best:
        return 0.3
    if opp_best > my_best:
        return -0.3
    return 0.0


# ---------------------------------------------------------------------------
# MCTS Node
# ---------------------------------------------------------------------------

class MCTSNode:
    __slots__ = ('move', 'parent', 'children', 'visits', 'value', 'prior', 'untried')

    def __init__(self, move, parent, prior: float = 1.0):
        self.move = move
        self.parent = parent
        self.children: dict = {}
        self.visits: int = 0
        self.value: float = 0.0   # from the perspective of the player who CHOSE this node
        self.prior: float = prior
        self.untried: Optional[list] = None

    def ucb_score(self, parent_visits: int) -> float:
        if self.visits == 0:
            return float('inf')
        q = self.value / self.visits
        u = C_PUCT * self.prior * math.sqrt(parent_visits) / (1 + self.visits)
        return q + u

    def best_child(self) -> 'MCTSNode':
        return max(self.children.values(), key=lambda n: n.visits)

    def select_child(self) -> 'MCTSNode':
        return max(self.children.values(), key=lambda n: n.ucb_score(self.visits))


# ---------------------------------------------------------------------------
# MCTS Engine
# ---------------------------------------------------------------------------

class MCTSEngine:

    def __init__(self, my: str, opp: str):
        self.my = my
        self.opp = opp
        self.root: Optional[MCTSNode] = None

    def reuse_or_reset(self, last_opp_move: Optional[Tuple[int, int]]):
        if self.root is not None and last_opp_move is not None:
            if last_opp_move in self.root.children:
                self.root = self.root.children[last_opp_move]
                self.root.parent = None
                print(f"  [MCTS] Tree reuse: root={last_opp_move} "
                      f"({self.root.visits} visits retained)")
                return
        self.root = MCTSNode(move=None, parent=None, prior=1.0)
        print("  [MCTS] Fresh tree")

    def advance_root(self, move: Tuple[int, int]):
        if move in self.root.children:
            self.root = self.root.children[move]
            self.root.parent = None
        else:
            self.root = MCTSNode(move=move, parent=None, prior=1.0)

    def search(self, board: dict, deadline: float) -> Tuple[int, int]:
        root = self.root
        bbox = compute_bbox(board)
        candidates = get_candidates(board, bbox)
        if not candidates:
            return 0, 0

        priors = compute_priors(board, candidates, self.my, self.opp)
        ordered = sorted(candidates, key=lambda m: priors[m], reverse=True)[:MAX_CANDIDATES]
        root.untried = list(ordered)
        for m in ordered:
            if m not in root.children:
                root.children[m] = MCTSNode(move=m, parent=root, prior=priors[m])

        iterations = 0
        while time.time() < deadline:
            self._iterate(root, dict(board), is_my_turn=True)
            iterations += 1

        best = root.best_child()
        wr = best.value / best.visits if best.visits else 0
        print(f"  [MCTS] {iterations} iters — best={best.move} "
              f"visits={best.visits} winrate={wr:.3f}")
        return best.move

    # ------------------------------------------------------------------

    def _iterate(self, root: MCTSNode, board: dict, is_my_turn: bool):
        node = root
        turn = is_my_turn
        path = []   # list of (node, turn_when_node_was_chosen)

        # --- Selection: walk until we find an unvisited or unexpanded node ---
        while node.untried is not None and len(node.untried) == 0 and node.children:
            node = node.select_child()
            player = self.my if turn else self.opp
            board[node.move] = player
            path.append((node, turn))
            turn = not turn

        # --- Expansion ---
        expanded_move = None
        if node.untried:
            move = node.untried.pop(0)
            player = self.my if turn else self.opp
            board[move] = player
            expanded_move = move

            prior = node.children[move].prior if move in node.children else 1.0
            child = node.children.get(move) or MCTSNode(move=move, parent=node, prior=prior)
            node.children[move] = child

            # Check win immediately — no need to rollout
            if is_winning_move(board, move, player):
                result = 1.0 if player == self.my else -1.0
                self._backprop(root, path, child, turn, result)
                # Undo
                del board[move]
                for n, _ in path:
                    if n.move in board:
                        del board[n.move]
                return

            bbox = compute_bbox(board)
            cands = get_candidates(board, bbox)
            if cands:
                cp = compute_priors(board, cands,
                                    self.my if turn else self.opp,
                                    self.opp if turn else self.my)
                child_ordered = sorted(cands, key=lambda m: cp[m], reverse=True)[:MAX_CANDIDATES]
                child.untried = list(child_ordered)
                for m2 in child_ordered:
                    if m2 not in child.children:
                        child.children[m2] = MCTSNode(move=m2, parent=child, prior=cp[m2])
            else:
                child.untried = []

            path.append((child, turn))
            turn = not turn
            node = child

        # --- Rollout ---
        result = self._rollout(board, turn)

        # --- Backpropagation ---
        self._backprop(root, path, node, turn, result)

        # Undo board changes from selection + expansion
        if expanded_move and expanded_move in board:
            del board[expanded_move]
        for n, _ in path[:-1] if expanded_move else path:
            if n.move and n.move in board:
                del board[n.move]

    def _backprop(self, root: MCTSNode, path: list, leaf: MCTSNode,
                  leaf_turn: bool, result: float):
        """
        result = 1.0 means MY player won.
        Each node stores value from the perspective of the player whose turn
        it was when that node was selected (i.e. the player who placed the move).
        We flip sign at each level walking back up.
        leaf_turn = whose turn it is NOW (after the leaf move was placed),
        so the leaf move was placed by (not leaf_turn).
        """
        # Value for the leaf node — placed by (not leaf_turn)
        leaf_placer_is_me = not leaf_turn  # True if my player placed the leaf move
        leaf.visits += 1
        leaf.value += result if leaf_placer_is_me else -result

        # Walk back up; each level flips sign because the player alternates
        v = -result if leaf_placer_is_me else result
        for node, node_turn in reversed(path):
            node.visits += 1
            node_placer_is_me = not node_turn
            node.value += v if node_placer_is_me else -v
            v = -v
        root.visits += 1

    def _rollout(self, board: dict, my_turn: bool) -> float:
        """
        Hybrid rollout:
          Plies 0..ADVERSARIAL_PLIES-1 — adversarial: each player samples from
            a softmax over their top-3 threat-scored moves, modelling realistic
            resistance rather than random play.
          Plies ADVERSARIAL_PLIES..MAX_ROLLOUT_DEPTH — stochastic biased rollout
            as before (cheaper, adds noise for exploration).
        """
        board = dict(board)
        turn = my_turn
        last_move = None

        for ply in range(MAX_ROLLOUT_DEPTH):
            bbox = compute_bbox(board, margin=2)
            player = self.my if turn else self.opp
            opponent = self.opp if turn else self.my
            cands = get_candidates(board, bbox)
            if not cands:
                break

            # Forced move check (both phases)
            forced = find_forced_move(board, cands, player, opponent)
            if forced:
                board[forced] = player
                last_move = forced
                if is_winning_move(board, forced, player):
                    return 1.0 if player == self.my else -1.0
                turn = not turn
                continue

            scores = [threat_prior(board, m, player, opponent) for m in cands]

            if ply < ADVERSARIAL_PLIES:
                # --- Adversarial phase: softmax over top-3 moves ---
                # Restricts to top 3 by threat score, then samples proportional
                # to softmax(score / temperature) so the opponent doesn't always
                # play the single best move but is heavily biased toward good ones.
                top_indices = sorted(range(len(cands)),
                                     key=lambda i: scores[i], reverse=True)[:3]
                top_cands  = [cands[i] for i in top_indices]
                top_scores = [scores[i] for i in top_indices]
                # Softmax with temperature
                max_s = max(top_scores)
                exp_s = [math.exp((s - max_s) / SOFTMAX_TEMP) for s in top_scores]
                total_exp = sum(exp_s)
                threshold = random.random() * total_exp
                cumulative = 0.0
                move = top_cands[-1]
                for m, e in zip(top_cands, exp_s):
                    cumulative += e
                    if cumulative >= threshold:
                        move = m
                        break
            else:
                # --- Stochastic phase: proportional sampling over all candidates ---
                total = sum(scores)
                threshold = random.random() * total
                cumulative = 0.0
                move = cands[-1]
                for m, s in zip(cands, scores):
                    cumulative += s
                    if cumulative >= threshold:
                        move = m
                        break

            board[move] = player
            last_move = move
            if is_winning_move(board, move, player):
                return 1.0 if player == self.my else -1.0
            turn = not turn

        if last_move:
            return fast_static_score(board, last_move, self.my, self.opp)
        return 0.0


# ---------------------------------------------------------------------------
# StudentAgent
# ---------------------------------------------------------------------------

class StudentAgent:

    def __init__(self):
        self.my_symbol: Optional[str] = None
        self.opp_symbol: Optional[str] = None
        self._engine: Optional[MCTSEngine] = None
        self._last_board: dict = {}

    def get_move(self, state: dict) -> Tuple[int, int]:
        grid_raw = state['grid']
        self.my_symbol = state['your_symbol']
        self.opp_symbol = state['opponent_symbol']
        time_remaining = float(state.get('time_remaining', 300))

        board = self._parse_grid(grid_raw)

        if not board:
            return 0, 0

        last_opp_move = self._find_new_move(self._last_board, board, self.opp_symbol)

        if self._engine is None:
            self._engine = MCTSEngine(self.my_symbol, self.opp_symbol)
        self._engine.my = self.my_symbol
        self._engine.opp = self.opp_symbol
        self._engine.reuse_or_reset(last_opp_move)

        bbox = compute_bbox(board)
        candidates = get_candidates(board, bbox)
        forced = find_forced_move(board, candidates, self.my_symbol, self.opp_symbol)
        if forced:
            print(f"  [Forced] {forced}")
            self._last_board = dict(board)
            self._engine.advance_root(forced)
            return forced

        budget = min(time_remaining * 0.85, 9.0)
        deadline = time.time() + budget

        move = self._engine.search(board, deadline)
        self._last_board = dict(board)
        self._engine.advance_root(move)
        return move

    @staticmethod
    def _find_new_move(old_board: dict, new_board: dict,
                       symbol: str) -> Optional[Tuple[int, int]]:
        for pos, sym in new_board.items():
            if sym == symbol and pos not in old_board:
                return pos
        return None

    @staticmethod
    def _parse_grid(grid_dict) -> dict:
        board = {}
        for key, value in grid_dict.items():
            try:
                row, col = eval(key) if isinstance(key, str) else key
                board[(row, col)] = value
            except Exception:
                pass
        return board


# ======================================================================
# Main menu
# ======================================================================

def main():
    SERVER_URL = "http://192.168.0.225:5000"
    CREDENTIALS_FILE = ".credentialsLocal"

    def load_credentials() -> dict:
        if os.path.exists(CREDENTIALS_FILE):
            try:
                with open(CREDENTIALS_FILE) as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def save_credentials(uname: str, player_id: str, api_token: str) -> None:
        with open(CREDENTIALS_FILE, "w") as f:
            json.dump({"username": uname, "player_id": player_id, "api_token": api_token}, f)
        os.chmod(CREDENTIALS_FILE, 0o600)

    client = GameClient(SERVER_URL)
    creds = load_credentials()

    print("=" * 60)
    print("Infinite Tic-Tac-Toe — Student Agent")
    print("=" * 60)

    if creds.get("username") and creds.get("api_token"):
        print(f"\nSaved credentials found for user '{creds['username']}'.")
        use_saved = input("Use saved credentials? [Y/n]: ").strip().lower()
        if use_saved in ("", "y", "yes"):
            client.player_id = creds["player_id"]
            client.username  = creds["username"]
            client.api_token = creds["api_token"]
            print(f"✓ Using saved credentials for '{client.username}'")
        else:
            creds = {}

    if not client.api_token:
        print("\n1 = Login to existing account")
        print("2 = Register a new account")
        auth_choice = input("Choice: ").strip()
        uname = input("Username: ").strip()
        password = input("Password: ").strip()
        try:
            if auth_choice == "1":
                client.login(uname, password)
            else:
                client.register(uname, password)
        except Exception as exc:
            print(f"\nAuth failed: {exc}")
            sys.exit(1)
        save_credentials(client.username, client.player_id, client.api_token)
        print(f"  Credentials saved to '{CREDENTIALS_FILE}'")

    agent = StudentAgent()

    while True:
        print("\n" + "=" * 60)
        print(f"Logged in as: {client.username}")
        print("Menu:")
        print("  1. Challenge AI (Ranked)")
        print("  2. Create Lobby (vs Player)")
        print("  3. Join Lobby (vs Player)")
        print("  4. View Leaderboards")
        print("  5. Play as Human (Web Interface)")
        print("  6. Exit")
        print("=" * 60)

        choice = input("\nChoice: ").strip()

        if choice == "1":
            print("\nChallenging AI...")
            game_id = client.challenge_ai()
            client.play_game(game_id, agent.get_move, verbose=True)

        elif choice == "2":
            print("\nCreating lobby...")
            raw = input("Max moves (default 200): ").strip()
            max_moves = int(raw) if raw else 200
            lobby_id = client.create_lobby(max_moves=max_moves)
            print(f"\nLobby ID: {lobby_id}")
            print("Share this ID with your opponent!")
            game_id = client.wait_for_lobby_start(lobby_id)
            client.play_game(game_id, agent.get_move, verbose=True)

        elif choice == "3":
            print("\nAvailable lobbies:")
            lobbies = client.list_lobbies()
            if not lobbies:
                print("No open lobbies available.")
                continue
            for i, lobby in enumerate(lobbies, 1):
                print(f"\n  {i}. Host: {lobby['host_name']}")
                print(f"     Max Moves: {lobby['max_moves']}")
                print(f"     Time Limit: {lobby['move_time_limit']}s")
                print(f"     Lobby ID: {lobby['lobby_id']}")
            lobby_num = input("\nEnter lobby number (or press Enter to enter ID): ").strip()
            if lobby_num == "":
                lobby_id = input("Enter lobby ID: ").strip()
            else:
                try:
                    lobby_id = lobbies[int(lobby_num) - 1]['lobby_id']
                except (ValueError, IndexError):
                    print("Invalid choice!")
                    continue
            game_id = client.join_lobby(lobby_id)
            client.play_game(game_id, agent.get_move, verbose=True)

        elif choice == "4":
            for table, label in (("ai", "AI Challenge"), ("pvp", "Player vs Player")):
                print(f"\n{'='*60}")
                print(f"Leaderboard — {label}")
                print(f"{'='*60}")
                try:
                    leaderboard = client.get_leaderboard(table)
                except Exception as exc:
                    print(f"  Could not fetch: {exc}")
                    continue
                if not leaderboard:
                    print("  No ranked games played yet.")
                    continue
                print(f"  {'Rank':<5} {'Player':<20} {'Score':<8} {'W-L-D':<12} {'Win%'}")
                print("  " + "-" * 55)
                for i, p in enumerate(leaderboard, 1):
                    total = p['total_games']
                    win_pct = (p['wins'] / total * 100) if total else 0.0
                    wld = f"{p['wins']}-{p['losses']}-{p['draws']}"
                    print(f"  {i:<5} {p['player_name']:<20} {p['score']:<8} {wld:<12} {win_pct:.1f}%")
            input("\nPress Enter to continue...")

        elif choice == "5":
            print("\nStarting AI challenge for the web interface...")
            game_id = client.challenge_ai()
            url = f"{SERVER_URL}/play?api_token={client.api_token}&game_id={game_id}"
            print(f"\nOpen this URL in your browser:")
            print(f"  {url}")
            input("\nPress Enter when done...")

        elif choice == "6":
            print("\nGoodbye!")
            break

        else:
            print("Invalid choice!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)