"""
Client Library for Infinite Tic-Tac-Toe

Interface for students:
- Register / login with username + password
- Create/join lobbies (game starts automatically)
- Challenge the built-in AI (ranked)
- Play games via polling loop
- View leaderboards
"""

import requests
import time
from typing import Callable, Optional, Tuple


class GameClient:
    """Client for the Infinite Tic-Tac-Toe tournament server."""

    def __init__(self, server_url: str = "https://inf-tac-toe.apps.dataintelligencelab.nl"):
        self.server_url = server_url.rstrip('/')
        self.player_id: Optional[str] = None
        self.username: Optional[str] = None
        self.api_token: Optional[str] = None

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    def register(self, username: str, password: str) -> str:
        """
        Register a new account.  Returns the api_token.

        The token is stored on self.api_token and used for all
        subsequent requests, so you do not need to pass it manually.
        """
        response = requests.post(
            f"{self.server_url}/api/auth/register",
            json={"username": username, "password": password},
        )
        response.raise_for_status()
        data = response.json()
        self.player_id = data['player_id']
        self.username = data['username']
        self.api_token = data['api_token']
        print(f"✓ Registered as '{self.username}'")
        print(f"  Player ID : {self.player_id}")
        print(f"  API Token : {self.api_token}")
        return self.api_token

    def login(self, username: str, password: str) -> str:
        """
        Log in to an existing account.  Returns the api_token.
        """
        response = requests.post(
            f"{self.server_url}/api/auth/login",
            json={"username": username, "password": password},
        )
        response.raise_for_status()
        data = response.json()
        self.player_id = data['player_id']
        self.username = data['username']
        self.api_token = data['api_token']
        print(f"✓ Logged in as '{self.username}'")
        return self.api_token

    def _auth(self) -> dict:
        """Return {'api_token': ...} for inclusion in request bodies."""
        if not self.api_token:
            raise ValueError("Not authenticated. Call register() or login() first.")
        return {"api_token": self.api_token}

    # ------------------------------------------------------------------
    # Lobby
    # ------------------------------------------------------------------

    def create_lobby(self, max_moves: int = 200, move_time_limit: int = 300) -> str:
        """Create a new lobby and return its lobby_id."""
        response = requests.post(
            f"{self.server_url}/api/lobbies/create",
            json={**self._auth(), "max_moves": max_moves, "move_time_limit": move_time_limit},
        )
        response.raise_for_status()
        lobby_id = response.json()['lobby_id']
        print(f"✓ Lobby created: {lobby_id}")
        print("  Waiting for opponent to join...")
        return lobby_id

    def list_lobbies(self) -> list:
        """Return a list of open lobbies."""
        response = requests.get(f"{self.server_url}/api/lobbies")
        response.raise_for_status()
        return response.json()['lobbies']

    def join_lobby(self, lobby_id: str) -> str:
        """
        Join an existing lobby.  The game starts immediately.
        Returns the game_id.
        """
        response = requests.post(
            f"{self.server_url}/api/lobbies/{lobby_id}/join",
            json=self._auth(),
        )
        response.raise_for_status()
        data = response.json()
        game_id = data['game_id']
        print(f"✓ Joined lobby — game started: {game_id}")
        return game_id

    def wait_for_lobby_start(self, lobby_id: str, poll_interval: float = 1.0) -> str:
        """
        Poll until someone joins the lobby and the game starts.
        Returns the game_id.
        """
        print("Waiting for opponent to join...")
        while True:
            response = requests.get(f"{self.server_url}/api/lobbies/{lobby_id}")
            if response.ok:
                lobby = response.json().get('lobby', {})
                if lobby.get('status') == 'started' and lobby.get('game_id'):
                    game_id = lobby['game_id']
                    print(f"✓ Game started: {game_id}")
                    return game_id
            time.sleep(poll_interval)

    # ------------------------------------------------------------------
    # AI Challenge
    # ------------------------------------------------------------------

    def challenge_ai(self, max_moves: int = 200, move_time_limit: int = 300) -> str:
        """Start a ranked game against the built-in AI. Returns the game_id."""
        response = requests.post(
            f"{self.server_url}/api/ai/challenge",
            json={**self._auth(), "max_moves": max_moves, "move_time_limit": move_time_limit},
        )
        response.raise_for_status()
        data = response.json()
        game_id = data['game_id']
        print(f"✓ AI challenge started: {game_id}")
        print(f"  You are: {data['your_symbol']}")
        return game_id

    # ------------------------------------------------------------------
    # Game play
    # ------------------------------------------------------------------

    def get_state(self, game_id: str) -> dict:
        """Return the current game state for the authenticated player."""
        if not self.api_token:
            raise ValueError("Not authenticated.")
        response = requests.get(
            f"{self.server_url}/api/games/{game_id}/state",
            params={"api_token": self.api_token},
        )
        response.raise_for_status()
        return response.json()

    def make_move(self, game_id: str, row: int, col: int) -> dict:
        """Submit a move. Returns the server response dict."""
        response = requests.post(
            f"{self.server_url}/api/games/{game_id}/move",
            json={**self._auth(), "row": row, "col": col},
        )
        if response.status_code == 400:
            return response.json()
        response.raise_for_status()
        return response.json()

    def get_board(self, game_id: str) -> str:
        """Return an ASCII board representation."""
        response = requests.get(f"{self.server_url}/api/games/{game_id}/board")
        response.raise_for_status()
        return response.json()['board']

    def get_leaderboard(self, table: str = 'ai') -> list:
        """
        Return leaderboard entries sorted by score.
        table: 'ai' (AI-challenge) or 'pvp' (player-vs-player).
        """
        url = f"{self.server_url}/api/leaderboard"
        if table in ('ai', 'pvp'):
            url += f"/{table}"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()['leaderboard']

    def get_game_history(self, game_id: str) -> list:
        """Return the full move history for a game."""
        response = requests.get(f"{self.server_url}/api/games/{game_id}/history")
        response.raise_for_status()
        return response.json()['moves']

    # ------------------------------------------------------------------
    # Game loop
    # ------------------------------------------------------------------

    def play_game(
        self,
        game_id: str,
        move_function: Callable[[dict], Tuple[int, int]],
        poll_interval: float = 1.0,
        verbose: bool = True,
    ) -> None:
        """
        Run the full game loop, calling move_function(state) on every turn.

        move_function receives the state dict and must return (row, col).
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Playing game {game_id}")
            print(f"{'='*60}")

        while True:
            state = self.get_state(game_id)

            if state['game_status'] != 'active':
                if verbose:
                    print(f"\n{self.get_board(game_id)}")
                    print(f"\n{'='*60}")
                    winner = state.get('winner')
                    if winner == state['your_symbol']:
                        print(f"You won as {state['your_symbol']}!")
                    elif winner == 'DRAW':
                        print("Game ended in a draw.")
                    elif winner:
                        print(f"You lost. {winner} won.")
                    if state['game_status'] == 'timeout':
                        print("Game ended by timeout.")
                    print(f"{'='*60}\n")
                break

            if not state['your_turn']:
                if verbose:
                    tl = int(state.get('time_remaining', 0))
                    print(f"  Waiting for opponent… (move {state['move_count']}, "
                          f"time: {tl // 60}:{tl % 60:02d})")
                time.sleep(poll_interval)
                continue

            if verbose:
                tl = int(state.get('time_remaining', 0))
                print(f"\nYour turn! (You are {state['your_symbol']}, "
                      f"move {state['move_count'] + 1}, "
                      f"time: {tl // 60}:{tl % 60:02d})")

            try:
                row, col = move_function(state)
                if verbose:
                    print(f"  Submitting move: ({row}, {col})")
                result = self.make_move(game_id, row, col)
                if not result.get('success'):
                    err = result.get('error', 'Unknown error')
                    print(f"  ERROR: {err}")
                    if err == 'Not your turn':
                        time.sleep(poll_interval)
                        continue
                    break
                if verbose:
                    print("  ✓ Move accepted")
                    try:
                        print(f"\n{self.get_board(game_id)}")
                    except Exception:
                        pass
            except Exception as exc:
                print(f"  ERROR in move function: {exc}")
                break

            time.sleep(0.3)
