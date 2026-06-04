"""
Student Agent for Infinite Tic-Tac-Toe

Algorithm: Iterative-Deepening Negamax with Alpha-Beta Pruning (v2)
- Adversarial game-tree search with alpha-beta cutoffs
- INCREMENTAL sliding-window evaluation (O(144) per node vs full-board scan)
- Double-threat (fork) detection via incrementally maintained threat sets
- Principal Variation Search (PVS) for reduced node expansion
- Dynamic candidate generation restricted to cells near existing pieces
- Move ordering: TT best move -> killer moves -> threat-scored candidates
- Zobrist-hashed transposition table
- Forced-move pre-check (instant win / block)
- Time-managed iterative deepening (always keeps a fallback move)
"""

from client import GameClient
from typing import Tuple, Optional, List, Dict
import json
import math
import os
import random
import sys
import time

# Game constants
WIN = 6
DIRECTIONS = [(0, 1), (1, 0), (1, 1), (1, -1)]

# Search parameters
MAX_CANDIDATES = 15       # branching factor cap per node
TIME_FRACTION  = 0.85     # use at most this share of remaining clock
TIME_CAP       = 12.0      # hard ceiling per move (seconds)

# Evaluation weights (index = own piece count in a window of 6)
ATTACK = [0, 1, 12, 120,  1_200,  120_000, 10_000_000]
DEFEND = [0, 2, 15, 150,  1_500,  150_000, 10_000_000]
FORK_BONUS = 500_000      # bonus for unblockable double-threat (fork)

# Convergence bonus: reward building threats in multiple directions
# If a piece contributes to strong threats (3+) in N directions,
# bonus = CONVERGE[N].  Encourages setting up multi-line attacks.
CONVERGE = [0, 0, 5_000, 20_000, 80_000]  # index = number of threatening directions

# Transposition-table flags
EXACT, LOWER, UPPER = 0, 1, 2


class _Timeout(Exception):
    """Raised when the search budget expires."""



#  Zobrist hashing (lazily generated, XOR-incremental)
_ZT: Dict[tuple, int] = {}


def _zob(pos: Tuple[int, int], sym: str) -> int:
    key = (pos, sym)
    v = _ZT.get(key)
    if v is None:
        v = random.getrandbits(64)
        _ZT[key] = v
    return v


def board_hash(board: dict) -> int:
    h = 0
    for pos, sym in board.items():
        h ^= _zob(pos, sym)
    return h


 
#  Utility helpers
 

def is_winning_move(board: dict, move: Tuple[int, int], player: str) -> bool:
    """True if placing *player* at *move* creates >= 6 in a row."""
    r, c = move
    for dr, dc in DIRECTIONS:
        cnt = 1
        for i in range(1, WIN):
            if board.get((r + dr * i, c + dc * i)) == player:
                cnt += 1
            else:
                break
        for i in range(1, WIN):
            if board.get((r - dr * i, c - dc * i)) == player:
                cnt += 1
            else:
                break
        if cnt >= WIN:
            return True
    return False


def get_candidates(board: dict, radius: int = 1) -> List[Tuple[int, int]]:
    """Empty cells within *radius* of any occupied cell."""
    occupied = set(board)
    seen: set = set()
    out: list = []
    for r, c in occupied:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if dr == 0 and dc == 0:
                    continue
                p = (r + dr, c + dc)
                if p not in occupied and p not in seen:
                    seen.add(p)
                    out.append(p)
    return out


def _threat_score(board: dict, move: Tuple[int, int],
                  player: str, opponent: str) -> float:
    """Quick per-move score used for candidate ordering."""
    r, c = move
    sc = 0.0
    for dr, dc in DIRECTIONS:
        my = opp = 0
        for i in range(1, WIN):
            s = board.get((r + dr * i, c + dc * i))
            if s == player:
                my += 1
            else:
                break
        for i in range(1, WIN):
            s = board.get((r - dr * i, c - dc * i))
            if s == player:
                my += 1
            else:
                break
        for i in range(1, WIN):
            s = board.get((r + dr * i, c + dc * i))
            if s == opponent:
                opp += 1
            else:
                break
        for i in range(1, WIN):
            s = board.get((r - dr * i, c - dc * i))
            if s == opponent:
                opp += 1
            else:
                break
        sc += my * my * 2 + opp * opp * 3
    return sc


 
#  Alpha-Beta engine  (negamax + PVS + incremental eval + fork detection)
 

class AlphaBetaEngine:

    def __init__(self):
        self.tt: dict = {}                       # zobrist -> (depth, score, flag, best_move)
        self.killers: Dict[int, list] = {}       # depth  -> [move, move]
        self.nodes = 0
        self.depth_reached = 0
        self.deadline = 0.0
        self._best: Optional[Tuple[int, int]] = None
        # Fixed symbols for the current game turn (set in choose())
        self._my: str = ''
        self._opp: str = ''
        # Threat sets: cells where placing a piece completes WIN-in-a-row
        self._my_threats: set = set()
        self._opp_threats: set = set()

    # ── initialisation (once per turn) ──────────────────────────────

    def _init_eval_and_threats(self, board: dict) -> float:
        """
        Full sliding-window evaluation from self._my's perspective.
        Also populates self._my_threats / self._opp_threats.
        Called once at the start of each turn.
        """
        self._my_threats.clear()
        self._opp_threats.clear()
        score = 0.0
        visited: set = set()
        my, opp = self._my, self._opp
        for pos in board:
            r, c = pos
            for di, (dr, dc) in enumerate(DIRECTIONS):
                for off in range(WIN):
                    sr, sc = r - dr * off, c - dc * off
                    wk = (sr, sc, di)
                    if wk in visited:
                        continue
                    visited.add(wk)
                    mc = oc = 0
                    empties: list = []
                    for i in range(WIN):
                        p = (sr + dr * i, sc + dc * i)
                        cell = board.get(p)
                        if cell == my:
                            mc += 1
                        elif cell == opp:
                            oc += 1
                        else:
                            empties.append(p)
                    if mc and not oc:
                        score += ATTACK[mc]
                        if mc == WIN - 1 and len(empties) == 1:
                            self._my_threats.add(empties[0])
                    elif oc and not mc:
                        score -= DEFEND[oc]
                        if oc == WIN - 1 and len(empties) == 1:
                            self._opp_threats.add(empties[0])
        return score

    # ── incremental make / unmake ──────────────────────────────────

    def _make(self, board: dict, move: Tuple[int, int],
             sym: str, zhash: int):
        """
        Place *sym* at *move*.  Returns (new_zhash, eval_delta, threat_changes).
        eval_delta is from self._my's perspective.
        Only the 24 windows through *move* are re-evaluated (O(144) lookups).
        """
        board[move] = sym
        new_zhash = zhash ^ _zob(move, sym)
        r, c = move
        delta = 0.0
        changes: list = []          # list of (code, cell) for undo
        my, opp = self._my, self._opp
        is_my = (sym == my)
        # Track best threat count per direction for convergence bonus
        my_threat_dirs = 0          # directions where sym==my has 3+ in a pure window
        opp_threat_dirs = 0         # directions where sym==opp has 3+ in a pure window

        for dr, dc in DIRECTIONS:
            best_my_in_dir = 0
            best_opp_in_dir = 0

            for off in range(WIN):
                sr, sc = r - dr * off, c - dc * off
                mc = oc = 0
                empties: list = []
                for i in range(WIN):
                    p = (sr + dr * i, sc + dc * i)
                    cell = board.get(p)
                    if cell == my:
                        mc += 1
                    elif cell == opp:
                        oc += 1
                    else:
                        empties.append(p)

                # New window score
                nw = ATTACK[mc] if (mc and not oc) else (-DEFEND[oc] if (oc and not mc) else 0)

                # Old window (before this piece was placed)
                om = mc - 1 if is_my else mc
                oo = oc if is_my else oc - 1
                ow = ATTACK[om] if (om > 0 and oo == 0) else (-DEFEND[oo] if (oo > 0 and om == 0) else 0)

                delta += nw - ow

                # Track best pure-window count in this direction
                if mc > 0 and oc == 0:
                    best_my_in_dir = max(best_my_in_dir, mc)
                if oc > 0 and mc == 0:
                    best_opp_in_dir = max(best_opp_in_dir, oc)

                # --- threat tracking ---
                # Old threats removed (window had WIN-1 + 1 empty before move)
                old_empties_len = len(empties) + 1   # move was empty before
                if om == WIN - 1 and oo == 0 and old_empties_len == 1:
                    t = empties[0] if empties else move
                    if t in self._my_threats:
                        self._my_threats.discard(t)
                        changes.append(('m', t))
                if oo == WIN - 1 and om == 0 and old_empties_len == 1:
                    t = empties[0] if empties else move
                    if t in self._opp_threats:
                        self._opp_threats.discard(t)
                        changes.append(('o', t))

                # New threats created
                if mc == WIN - 1 and oc == 0 and len(empties) == 1:
                    t = empties[0]
                    if t not in self._my_threats:
                        self._my_threats.add(t)
                        changes.append(('M', t))
                if oc == WIN - 1 and mc == 0 and len(empties) == 1:
                    t = empties[0]
                    if t not in self._opp_threats:
                        self._opp_threats.add(t)
                        changes.append(('O', t))

            # Count this direction if it has a strong threat (3+ pieces)
            if best_my_in_dir >= 3:
                my_threat_dirs += 1
            if best_opp_in_dir >= 3:
                opp_threat_dirs += 1

        # Convergence bonus: reward building threats in multiple directions
        if my_threat_dirs >= 2:
            delta += CONVERGE[min(my_threat_dirs, 4)] * (1 if is_my else -1)
        if opp_threat_dirs >= 2:
            delta += CONVERGE[min(opp_threat_dirs, 4)] * (-1 if is_my else 1)

        return new_zhash, delta, changes

    def _unmake(self, board: dict, move: Tuple[int, int], changes: list):
        """Undo a move and restore threat sets."""
        del board[move]
        for code, cell in reversed(changes):
            if code == 'M':
                self._my_threats.discard(cell)
            elif code == 'm':
                self._my_threats.add(cell)
            elif code == 'O':
                self._opp_threats.discard(cell)
            elif code == 'o':
                self._opp_threats.add(cell)

    def _leaf_score(self, running_score: float, player: str) -> float:
        """
        Convert the absolute running score (from self._my's perspective)
        to the side-to-move's perspective, adding fork bonuses.
        """
        fork = 0.0
        if len(self._my_threats) >= 2:
            fork += FORK_BONUS
        if len(self._opp_threats) >= 2:
            fork -= FORK_BONUS
        total = running_score + fork
        return total if player == self._my else -total

    # ── public entry ────────────────────────────────────────────────

    def choose(self, board: dict, my: str, opp: str,
               time_left: float) -> Tuple[int, int]:
        budget = min(time_left * TIME_FRACTION, TIME_CAP)
        self.deadline = time.time() + budget
        self.nodes = 0
        self.depth_reached = 0
        self.killers.clear()
        self._my = my
        self._opp = opp

        # Wider radius at root for better coverage
        cands = get_candidates(board, radius=2)
        if not cands:
            return (0, 0)

        # ── Forced moves (instant win / block) ──
        for m in cands:
            if is_winning_move(board, m, my):
                print(f"  [AB] Instant win: {m}")
                return m
        for m in cands:
            if is_winning_move(board, m, opp):
                print(f"  [AB] Forced block: {m}")
                return m

        # ── Sort & cap candidates ──
        cands.sort(key=lambda m: _threat_score(board, m, my, opp), reverse=True)
        cands = cands[:MAX_CANDIDATES]
        self._best = cands[0]

        zhash = board_hash(board)
        running_score = self._init_eval_and_threats(board)

        depth = 1
        try:
            while True:
                if time.time() >= self.deadline:
                    break
                n0 = self.nodes
                self._root(board, cands, depth, zhash, running_score)
                print(f"  [AB] d={depth}  nodes={self.nodes - n0}  "
                      f"best={self._best}")
                depth += 1
                # Don't start a new ply if >45 % of budget is gone
                if time.time() > self.deadline - budget * 0.55:
                    break
        except _Timeout:
            pass

        self.depth_reached = depth - 1
        print(f"  [AB] done  d={self.depth_reached}  total_nodes={self.nodes}  "
              f"move={self._best}")
        return self._best

    # ── root search with PVS ───────────────────────────────────────

    def _root(self, board, cands, depth, zhash, running_score):
        alpha, beta = -math.inf, math.inf
        best_sc = -math.inf
        best_mv = cands[0]
        ordered = self._order(cands, zhash, depth)
        my, opp = self._my, self._opp

        for idx, mv in enumerate(ordered):
            self._ck()
            nz, delta, tch = self._make(board, mv, my, zhash)
            new_rs = running_score + delta

            if is_winning_move(board, mv, my):
                self._unmake(board, mv, tch)
                self._best = mv
                self.tt[zhash] = (depth, 10_000_000, EXACT, mv)
                return

            # PVS: full window for first move, null window for rest
            if idx == 0:
                sc = -self._ab(board, depth - 1, -beta, -alpha,
                               opp, my, nz, new_rs)
            else:
                sc = -self._ab(board, depth - 1, -alpha - 1, -alpha,
                               opp, my, nz, new_rs)
                if alpha < sc < beta:
                    sc = -self._ab(board, depth - 1, -beta, -alpha,
                                   opp, my, nz, new_rs)

            self._unmake(board, mv, tch)
            if sc > best_sc:
                best_sc = sc
                best_mv = mv
            alpha = max(alpha, sc)
            if alpha >= beta:
                break

        self._best = best_mv
        self.tt[zhash] = (depth, best_sc, EXACT, best_mv)

    # ── negamax with PVS + incremental eval ─────────────────────────

    def _ab(self, board: dict, depth: int,
            alpha: float, beta: float,
            player: str, opponent: str,
            zhash: int, running_score: float) -> float:
        """
        Returns score from *player*'s perspective (player = side to move).
        running_score is the absolute board eval from self._my's perspective.
        """
        self.nodes += 1
        if self.nodes & 0x3FF == 0:          # time-check every 1024 nodes
            self._ck()

        # ── TT probe ──
        entry = self.tt.get(zhash)
        tt_mv = None
        if entry:
            ed, es, ef, em = entry
            tt_mv = em
            if ed >= depth:
                if ef == EXACT:
                    return es
                if ef == LOWER and es >= beta:
                    return es
                if ef == UPPER and es <= alpha:
                    return es

        # ── Leaf evaluation (incremental + fork bonus) ──
        if depth <= 0:
            return self._leaf_score(running_score, player)

        cands = get_candidates(board)        # radius 1 inside tree
        if not cands:
            return self._leaf_score(running_score, player)

        # ── Instant-win scan ──
        for m in cands:
            if is_winning_move(board, m, player):
                return 10_000_000 + depth    # prefer faster wins

        # ── Must-block detection ──
        threats = [m for m in cands if is_winning_move(board, m, opponent)]
        if len(threats) > 1:                 # two+ threats = unavoidable loss
            return -(10_000_000 + depth)
        if threats:
            cands = threats                  # forced to block the one threat
        else:
            cands.sort(key=lambda m: _threat_score(board, m, player, opponent),
                       reverse=True)
            cands = cands[:MAX_CANDIDATES]

        # ── Move ordering (TT move & killers first) ──
        ordered = self._order(cands, zhash, depth)
        if tt_mv and tt_mv in ordered:
            ordered.remove(tt_mv)
            ordered.insert(0, tt_mv)

        orig_alpha = alpha
        best_sc = -math.inf
        best_mv = ordered[0]

        for idx, mv in enumerate(ordered):
            nz, delta, tch = self._make(board, mv, player, zhash)
            new_rs = running_score + delta

            if is_winning_move(board, mv, player):
                self._unmake(board, mv, tch)
                best_sc = 10_000_000 + depth
                best_mv = mv
                break

            # PVS: full window for first move, null window for rest
            if idx == 0:
                sc = -self._ab(board, depth - 1, -beta, -alpha,
                               opponent, player, nz, new_rs)
            else:
                sc = -self._ab(board, depth - 1, -alpha - 1, -alpha,
                               opponent, player, nz, new_rs)
                if alpha < sc < beta:
                    sc = -self._ab(board, depth - 1, -beta, -alpha,
                                   opponent, player, nz, new_rs)

            self._unmake(board, mv, tch)
            if sc > best_sc:
                best_sc = sc
                best_mv = mv
            alpha = max(alpha, sc)
            if alpha >= beta:
                kl = self.killers.setdefault(depth, [])
                if mv not in kl:
                    kl.insert(0, mv)
                    if len(kl) > 2:
                        kl.pop()
                break

        # ── TT store ──
        if best_sc <= orig_alpha:
            flag = UPPER
        elif best_sc >= beta:
            flag = LOWER
        else:
            flag = EXACT
        self.tt[zhash] = (depth, best_sc, flag, best_mv)
        return best_sc

    # ── helpers ─────────────────────────────────────────────────────

    def _order(self, cands, zhash, depth):
        """Push TT best-move and killer moves to the front."""
        out = list(cands)
        for km in self.killers.get(depth, []):
            if km in out:
                out.remove(km)
                out.insert(0, km)
        entry = self.tt.get(zhash)
        if entry and entry[3] and entry[3] in out:
            out.remove(entry[3])
            out.insert(0, entry[3])
        return out

    def _ck(self):
        if time.time() >= self.deadline:
            raise _Timeout


 
#  StudentAgent  (public interface expected by client.py)
 

class StudentAgent:
    """Alpha-Beta agent for Infinite Tic-Tac-Toe."""

    def __init__(self):
        self.my_symbol = None
        self.opponent_symbol = None
        self._engine = AlphaBetaEngine()
        self.last_move_stats: Optional[dict] = None

    def get_move(self, state: dict) -> Tuple[int, int]:
        self.my_symbol = state['your_symbol']
        self.opponent_symbol = state['opponent_symbol']
        time_left = float(state.get('time_remaining', 300))
        board = self._parse_grid(state['grid'])
        if not board:
            self.last_move_stats = {"depth": 0, "nodes": 0,
                                    "time_remaining": time_left}
            return (0, 0)
        move = self._engine.choose(board, self.my_symbol,
                                   self.opponent_symbol, time_left)
        self.last_move_stats = {
            "depth": self._engine.depth_reached,
            "nodes": self._engine.nodes,
            "time_remaining": time_left,
        }
        return move

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


 
#  Game Logger
 

class GameLogger:
    """
    Logs game outcomes, move histories, and per-move search stats
    to a SQLite database (game_logs.db).
    """

    def __init__(self):
        from game_db import GameDB
        self._db = GameDB()
        self._game_id: Optional[str] = None
        self._timestamp: str = ""
        self._game_type: str = ""
        self._agent_stats: list = []

    # ── per-game lifecycle ────────────────────────────────────────

    def start_game(self, game_id: str, game_type: str,
                   agent_name: str = "AlphaBeta-v2",
                   opponent_name: str = ""):
        """Call before play_game().  game_type = 'ai' | 'pvp'."""
        self._game_id = game_id
        self._timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        self._game_type = game_type
        self._agent_name = agent_name
        self._opponent_name = opponent_name
        self._agent_stats = []

    def log_move(self, move_number: int, move: Tuple[int, int], stats: dict):
        """Record one agent move with its search statistics."""
        self._agent_stats.append({
            "move_number": move_number,
            "move": move,
            "depth_reached": stats.get("depth", 0),
            "nodes_searched": stats.get("nodes", 0),
            "time_remaining": round(stats.get("time_remaining", 0), 1),
        })

    def finish_game(self, client: GameClient, game_id: str):
        """Call after play_game().  Fetches result & history, saves to DB."""
        if not self._game_id:
            return

        my_sym = opp_sym = "?"
        result = "error"
        total_moves = 0

        # Fetch final game state
        try:
            state = client.get_state(game_id)
            my_sym = state.get("your_symbol", "?")
            opp_sym = state.get("opponent_symbol", "?")
            winner = state.get("winner")
            status = state.get("game_status", "")
            total_moves = state.get("move_count", 0)

            if winner == my_sym:
                result = "win"
            elif winner == "DRAW" or status == "draw":
                result = "draw"
            elif status == "timeout":
                result = "timeout"
            else:
                result = "loss"
        except Exception:
            pass

        # Detect opponent name from move history if not set
        if not self._opponent_name:
            try:
                hist = client.get_game_history(game_id)
                for m in hist:
                    if m.get("player_name") and m["player_name"] != client.username:
                        self._opponent_name = m["player_name"]
                        break
            except Exception:
                pass

        # Insert game record
        self._db.insert_game(
            game_id=game_id, timestamp=self._timestamp,
            game_type=self._game_type, my_symbol=my_sym,
            opponent_symbol=opp_sym, result=result,
            total_moves=total_moves,
            agent_name=self._agent_name,
            opponent_name=self._opponent_name,
        )

        # Insert move history from server
        try:
            for m in client.get_game_history(game_id):
                self._db.insert_move(
                    game_id=game_id,
                    move_number=m.get("move_number", 0),
                    row=m.get("row", 0), col=m.get("col", 0),
                    player_id=m.get("player_id", ""),
                    player_name=m.get("player_name", ""),
                    played_at=m.get("played_at", ""),
                )
        except Exception:
            pass

        # Insert agent search stats
        for s in self._agent_stats:
            mv = s["move"]
            self._db.insert_agent_stat(
                game_id=game_id,
                move_number=s["move_number"],
                row=mv[0], col=mv[1],
                depth_reached=s["depth_reached"],
                nodes_searched=s["nodes_searched"],
                time_remaining=s["time_remaining"],
            )

        self._db.commit()

        # Print running tally
        summary = self._db.get_summary()
        print(f"\n  [LOG] Saved to game_logs.db — "
              f"Record: {summary['wins']}W–{summary['losses']}L–{summary['draws']}D  "
              f"({summary['total']} games total)")
        self._game_id = None


def main():
    """Main menu for students"""

    SERVER_URL = "http://192.168.0.225:5000"
    CREDENTIALS_FILE = ".credentialsLocal1"

    # ------------------------------------------------------------------ #
    # Credential helpers
    # ------------------------------------------------------------------ #

    def load_credentials() -> dict:
        if os.path.exists(CREDENTIALS_FILE):
            try:
                with open(CREDENTIALS_FILE) as f:
                    return json.load(f)
            except Exception:
                pass
        return {}

    def save_credentials(username: str, player_id: str, api_token: str) -> None:
        with open(CREDENTIALS_FILE, "w") as f:
            json.dump({"username": username, "player_id": player_id, "api_token": api_token}, f)
        os.chmod(CREDENTIALS_FILE, 0o600)

    # ------------------------------------------------------------------ #
    # Auth flow
    # ------------------------------------------------------------------ #

    client = GameClient(SERVER_URL)
    saved = load_credentials()

    print("=" * 60)
    print("Infinite Tic-Tac-Toe — Student Agent")
    print("=" * 60)

    if saved.get("username") and saved.get("api_token"):
        print(f"\nSaved credentials found for user '{saved['username']}'.")
        use_saved = input("Use saved credentials? [Y/n]: ").strip().lower()
        if use_saved in ("", "y", "yes"):
            client.player_id = saved["player_id"]
            client.username = saved["username"]
            client.api_token = saved["api_token"]
            print(f"✓ Using saved credentials for '{client.username}'")
        else:
            saved = {}

    if not client.api_token:
        print("\n1 = Login to existing account")
        print("2 = Register a new account")
        auth_choice = input("Choice: ").strip()
        username = input("Username: ").strip()
        password = input("Password: ").strip()
        try:
            if auth_choice == "1":
                client.login(username, password)
            else:
                client.register(username, password)
        except Exception as exc:
            print(f"\nAuth failed: {exc}")
            sys.exit(1)
        save_credentials(client.username, client.player_id, client.api_token)
        print(f"  Credentials saved to '{CREDENTIALS_FILE}'")

    agent = StudentAgent()
    logger = GameLogger()

    def logged_move(state):
        """Wrapper that logs search stats after every agent move."""
        move = agent.get_move(state)
        if agent.last_move_stats:
            logger.log_move(
                state.get('move_count', 0) + 1,
                move,
                agent.last_move_stats,
            )
        return move

    # ------------------------------------------------------------------ #
    # Main menu
    # ------------------------------------------------------------------ #

    while True:
        print("\n" + "=" * 60)
        print(f"Logged in as: {client.username}")
        print("Menu:")
        print("  1. Challenge AI (Ranked)")
        print("  2. Create Lobby (vs Player)")
        print("  3. Join Lobby (vs Player)")
        print("  4. View Leaderboards")
        print("  5. Play as Human (Web Interface)")
        print("  6. Batch AI Challenge (play N games)")
        print("  7. Exit")
        print("=" * 60)

        choice = input("\nChoice: ").strip()

        if choice == "1":
            print("\nChallenging AI...")
            game_id = client.challenge_ai()
            logger.start_game(game_id, "ai")
            client.play_game(game_id, logged_move, verbose=True)
            logger.finish_game(client, game_id)

        elif choice == "2":
            print("\nCreating lobby...")
            raw = input("Max moves (default 200): ").strip()
            max_moves = int(raw) if raw else 200

            lobby_id = client.create_lobby(max_moves=max_moves)
            print(f"\nLobby ID: {lobby_id}")
            print("Share this ID with your opponent!")

            game_id = client.wait_for_lobby_start(lobby_id)
            logger.start_game(game_id, "pvp")
            client.play_game(game_id, logged_move, verbose=True)
            logger.finish_game(client, game_id)

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
            logger.start_game(game_id, "pvp")
            client.play_game(game_id, logged_move, verbose=True)
            logger.finish_game(client, game_id)

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

                print(f"  {'Rank':<5} {'Player':<20} {'Score':<8} {'W–L–D':<12} {'Win%'}")
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
            raw = input("\nHow many AI games? ").strip()
            num_games = int(raw) if raw else 15
            print(f"\n{'='*60}")
            print(f"Batch: playing {num_games} AI games")
            print(f"{'='*60}")
            for gi in range(1, num_games + 1):
                try:
                    print(f"\n--- Game {gi}/{num_games} ---")
                    game_id = client.challenge_ai()
                    logger.start_game(game_id, "ai")
                    client.play_game(game_id, logged_move, verbose=True)
                    logger.finish_game(client, game_id)
                except KeyboardInterrupt:
                    print(f"\nBatch interrupted after {gi - 1} games.")
                    break
                except Exception as exc:
                    print(f"  Game {gi} error: {exc}")
                    continue
            print(f"\nBatch complete.")

        elif choice == "7":
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