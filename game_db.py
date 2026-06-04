"""
SQLite database for Infinite Tic-Tac-Toe game logs.

Schema:
  games           — one row per game (outcome, metadata)
  move_history    — full move list from the server/engine (both players)
  agent_stats     — per-move search stats (depth, nodes, time) for our agent

Usage:
    from game_db import GameDB
    db = GameDB()                         # opens/creates game_logs.db
    game_id = db.insert_game(...)         # returns the DB row id
    db.insert_move(game_id, ...)
    db.insert_agent_stat(game_id, ...)
    games = db.get_all_games()            # list of dicts
    stats = db.get_agent_stats(game_id)   # list of dicts
"""

import sqlite3
import json
import os
from typing import Optional

DB_FILE = "game_logs.db"


class GameDB:

    def __init__(self, path: str = DB_FILE):
        self.path = path
        self.conn = sqlite3.connect(path, timeout=30)
        self.conn.row_factory = sqlite3.Row  # dict-like access
        self.conn.execute("PRAGMA journal_mode=WAL")  # better concurrency
        self._create_tables()

    def _create_tables(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS games (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id       TEXT UNIQUE,
                timestamp     TEXT,
                type          TEXT,        -- 'ai' | 'pvp' | 'selfplay'
                my_symbol     TEXT,
                opponent_symbol TEXT,
                result        TEXT,        -- 'win' | 'loss' | 'draw' | 'timeout' | 'error'
                total_moves   INTEGER,
                agent_name    TEXT DEFAULT '',  -- e.g. 'AlphaBeta-v2', 'MCTS-PUCT'
                opponent_name TEXT DEFAULT '',  -- e.g. 'ai_strong', 'Joel-MCTS'
                config_notes  TEXT DEFAULT ''   -- optional: weight config, agent version, etc.
            );

            CREATE TABLE IF NOT EXISTS move_history (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id       TEXT,
                move_number   INTEGER,
                row           INTEGER,
                col           INTEGER,
                player_id     TEXT,
                player_name   TEXT,
                played_at     TEXT,
                FOREIGN KEY (game_id) REFERENCES games(game_id)
            );

            CREATE TABLE IF NOT EXISTS agent_stats (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                game_id       TEXT,
                move_number   INTEGER,
                row           INTEGER,
                col           INTEGER,
                depth_reached INTEGER,
                nodes_searched INTEGER,
                time_remaining REAL,
                FOREIGN KEY (game_id) REFERENCES games(game_id)
            );

            CREATE INDEX IF NOT EXISTS idx_move_history_game ON move_history(game_id);
            CREATE INDEX IF NOT EXISTS idx_agent_stats_game ON agent_stats(game_id);
        """)
        self.conn.commit()

    # ── Insert operations ───────────────────────────────────────────

    def insert_game(self, game_id: str, timestamp: str, game_type: str,
                    my_symbol: str, opponent_symbol: str, result: str,
                    total_moves: int, agent_name: str = "",
                    opponent_name: str = "", config_notes: str = "") -> str:
        self.conn.execute("""
            INSERT OR REPLACE INTO games
            (game_id, timestamp, type, my_symbol, opponent_symbol, result,
             total_moves, agent_name, opponent_name, config_notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (game_id, timestamp, game_type, my_symbol, opponent_symbol,
              result, total_moves, agent_name, opponent_name, config_notes))
        self.conn.commit()
        return game_id

    def insert_move(self, game_id: str, move_number: int, row: int, col: int,
                    player_id: str = "", player_name: str = "",
                    played_at: str = ""):
        self.conn.execute("""
            INSERT INTO move_history
            (game_id, move_number, row, col, player_id, player_name, played_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (game_id, move_number, row, col, player_id, player_name, played_at))

    def insert_agent_stat(self, game_id: str, move_number: int,
                          row: int, col: int, depth_reached: int,
                          nodes_searched: int, time_remaining: float):
        self.conn.execute("""
            INSERT INTO agent_stats
            (game_id, move_number, row, col, depth_reached, nodes_searched, time_remaining)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (game_id, move_number, row, col, depth_reached, nodes_searched,
              time_remaining))

    def commit(self):
        self.conn.commit()

    # ── Query operations ────────────────────────────────────────────

    def get_all_games(self, game_type: Optional[str] = None) -> list:
        """Return all games as a list of dicts."""
        if game_type:
            rows = self.conn.execute(
                "SELECT * FROM games WHERE type = ? ORDER BY id", (game_type,)
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM games ORDER BY id"
            ).fetchall()
        return [dict(r) for r in rows]

    def get_move_history(self, game_id: str) -> list:
        rows = self.conn.execute(
            "SELECT * FROM move_history WHERE game_id = ? ORDER BY move_number",
            (game_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_agent_stats(self, game_id: str) -> list:
        rows = self.conn.execute(
            "SELECT * FROM agent_stats WHERE game_id = ? ORDER BY move_number",
            (game_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def get_summary(self, game_type: Optional[str] = None) -> dict:
        """Quick summary: total, wins, losses, draws."""
        where = "WHERE type = ?" if game_type else ""
        params = (game_type,) if game_type else ()
        row = self.conn.execute(f"""
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN result='win' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN result='loss' THEN 1 ELSE 0 END) as losses,
                SUM(CASE WHEN result='draw' THEN 1 ELSE 0 END) as draws,
                SUM(CASE WHEN result='timeout' THEN 1 ELSE 0 END) as timeouts
            FROM games {where}
        """, params).fetchone()
        return dict(row)

    # ── Migration ───────────────────────────────────────────────────

    def migrate_from_json(self, json_path: str = "game_logs.json") -> int:
        """Import games from the old JSON log file. Returns count imported."""
        if not os.path.exists(json_path):
            return 0

        with open(json_path) as f:
            data = json.load(f)

        count = 0
        for g in data.get("games", []):
            game_id = g.get("game_id", "")
            # Skip if already imported
            existing = self.conn.execute(
                "SELECT 1 FROM games WHERE game_id = ?", (game_id,)
            ).fetchone()
            if existing:
                continue

            self.insert_game(
                game_id=game_id,
                timestamp=g.get("timestamp", ""),
                game_type=g.get("type", "unknown"),
                my_symbol=g.get("my_symbol", "?"),
                opponent_symbol=g.get("opponent_symbol", "?"),
                result=g.get("result", "unknown"),
                total_moves=g.get("total_moves", 0),
            )

            # Move history
            for m in g.get("move_history", []):
                self.insert_move(
                    game_id=game_id,
                    move_number=m.get("move_number", 0),
                    row=m.get("row", 0),
                    col=m.get("col", 0),
                    player_id=m.get("player_id", ""),
                    player_name=m.get("player_name", ""),
                    played_at=m.get("played_at", ""),
                )

            # Agent stats
            for s in g.get("agent_search_stats", []):
                move = s.get("move", [0, 0])
                self.insert_agent_stat(
                    game_id=game_id,
                    move_number=s.get("move_number", 0),
                    row=move[0] if isinstance(move, list) else 0,
                    col=move[1] if isinstance(move, list) else 0,
                    depth_reached=s.get("depth_reached", 0),
                    nodes_searched=s.get("nodes_searched", 0),
                    time_remaining=s.get("time_remaining", 0),
                )

            count += 1

        self.commit()
        return count

    def close(self):
        self.conn.close()


# ── CLI: run directly to migrate ────────────────────────────────────

if __name__ == "__main__":
    db = GameDB()
    n = db.migrate_from_json()
    if n:
        print(f"Migrated {n} games from game_logs.json -> {DB_FILE}")
    else:
        print("No new games to migrate (already imported or no JSON file)")
    s = db.get_summary()
    print(f"Database: {s['total']} games  "
          f"({s['wins']}W-{s['losses']}L-{s['draws']}D)")
    db.close()
