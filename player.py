from __future__ import annotations

import random
import copy
from functools import lru_cache
from typing import Optional, List, Tuple, Dict

import torch
import torch.nn.functional as F

import chess
from transformers import AutoTokenizer, AutoModelForCausalLM

from chess_tournament import Player

torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

def _device() -> torch.device:
  if torch.cuda.is_available():
    return torch.device("cuda")
  return torch.device("cpu")

class TransformerPlayer(Player):
  """
  TransformerPlayer (V1)

  Strategy:
    1) Opening book for a few early positions (exact FEN -> UCI move).
    2) Tactical safety layer:
         - Play mate-in-1 if available.
         - Avoid moves that allow opponent mate-in-1.
    3) Policy selection using a general pretrained decoder LM:
         - Score legal moves by average log-probability under the LM given a FEN prompt.
         - Use a light 2-ply "minimax-lite" over the top-K LM moves:
             pick move maximizing (my_score - opponent_best_reply_score).
    4) Speed:
         - KV-cache reuse: process prompt once, then score move tokens cheaply.
         - Caching: memoize (fen, move-set) -> scores via lru_cache.
  """

  def __init__(self, name: str):
    super().__init__(name)

    # ----------------------------
    # Model + tokenizer (local, transformer-based)
    # ----------------------------
    self.model_name = "distilgpt2"
    self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

    # GPT2-family has no pad token by default; use EOS for batching/padding.
    if self.tokenizer.pad_token is None:
      self.tokenizer.pad_token = self.tokenizer.eos_token

    self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
    self.model.eval()

    self.dev = _device()
    self.model.to(self.dev)

    # Speed: fp16 inference on GPU
    if self.dev.type == "cuda":
      self.model.half()

    # ----------------------------
    # Caches
    # ----------------------------
    # NOTE: lru_cache size is set on the decorator for _cached_scores.
    self._score_cache_max = 4096

    # ----------------------------
    # Opening book (exact FEN -> UCI)
    # ----------------------------
    def build_opening_book() -> Dict[str, str]:
      """
      Returns: dict mapping exact FEN -> our chosen UCI move.

      The book is built from short move sequences. For every prefix position where it is
      "our turn" in that sequence, we record the next move as the book move.
      """
      book: Dict[str, str] = {}

      def add_line(moves: List[str]) -> None:
        """
        Add a full line of UCI moves starting from the initial position.
        Stores (fen_before_our_move -> our_move) for every step where it's our turn.
        """
        board = chess.Board()
        for uci in moves:
          fen = board.fen()
          # Only store if the move is legal in this position.
          mv = chess.Move.from_uci(uci)
          if mv in board.legal_moves:
            # If multiple lines write the same FEN, keep the first.
            book.setdefault(fen, uci)
            board.push(mv)
          else:
            # Stop line if illegal
            break

      # ------------------------------------------------------------------------
      # WHITE REPERTOIRE (trappy but not suicidal)
      # Core idea: e4, Nf3, Bc4, c3, d4 (Italian / Scotch-ish center breaks)
      # Plus responses to common + random early black moves.
      # ------------------------------------------------------------------------

      # Start: always play 1.e4
      add_line(["e2e4"])

      # If Black plays a normal response:
      # 1.e4 e5 2.Nf3 (then aim Bc4 and d4 ideas)
      add_line(["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "d2d3"])
      # Alternative: Scotch-style center break if you want more tactics:
      add_line(["e2e4", "e7e5", "g1f3", "b8c6", "d2d4", "e5d4", "f3d4"])

      # 1.e4 c5 (Sicilian): keep it simple and active
      add_line(["e2e4", "c7c5", "g1f3", "d7d6", "d2d4"])
      add_line(["e2e4", "c7c5", "g1f3", "e7e6", "d2d4"])
      add_line(["e2e4", "c7c5", "g1f3", "g8f6", "d2d4"])

      # 1.e4 e6 (French): classical setup
      add_line(["e2e4", "e7e6", "d2d4", "d7d5", "b1c3"])

      # 1.e4 c6 (Caro-Kann): classical setup
      add_line(["e2e4", "c7c6", "d2d4", "d7d5", "b1c3"])

      # 1.e4 d5 (Scandinavian): just develop (many bots play this)
      add_line(["e2e4", "d7d5", "e4d5", "d8d5", "b1c3"])

      # "Random-ish" black replies after 1.e4: a6, h6, g6, b6, f6, g5, d6, Nc6, Nf6
      # We respond with principled development: Nf3 and/or Bc4 and/or d4.
      for black_reply in ["a7a6", "h7h6", "g7g6", "b7b6", "f7f6", "g7g5", "d7d6", "b8c6", "g8f6"]:
        add_line(["e2e4", black_reply, "g1f3"])

      # If they play ...g6 early, punish with quick center + bishop pressure
      add_line(["e2e4", "g7g6", "d2d4", "f8g7", "f1c4"])

      # If they play ...f6 (awful), go for immediate center hit
      add_line(["e2e4", "f7f6", "d2d4"])

      # ------------------------------------------------------------------------
      # BLACK REPERTOIRE vs 1.e4 (Caro-Kann, solid)
      # Goal: c6 + d5 structure, develop naturally.
      # ------------------------------------------------------------------------

      # We need to add black lines too. To do that, we encode full sequences from the start,
      # where White plays e4 first, then we play our black response.

      # 1.e4 -> 1...c6
      add_line(["e2e4", "c7c6"])

      # Typical: 2.d4 -> 2...d5 -> 3.Nc3 -> 3...dxe4 (mainline)
      add_line(["e2e4", "c7c6", "d2d4", "d7d5", "b1c3", "d5e4"])

      # If White plays 2.Nf3 instead of 2.d4: still play 2...d5
      add_line(["e2e4", "c7c6", "g1f3", "d7d5"])

      # If White plays 2.Nc3: play 2...d5
      add_line(["e2e4", "c7c6", "b1c3", "d7d5"])

      # If White plays a random second move (Bc4, f4, h3, etc.), still go ...d5
      for white_second in ["f1c4", "f2f4", "h2h3", "a2a3", "g2g3", "b2b3"]:
        add_line(["e2e4", "c7c6", white_second, "d7d5"])

      # ------------------------------------------------------------------------
      # BLACK REPERTOIRE vs 1.d4 (solid: ...d5, then ...Nf6 / ...e6)
      # ------------------------------------------------------------------------

      add_line(["d2d4", "d7d5"])

      # If 2.c4 -> ...e6 (QGD structure)
      add_line(["d2d4", "d7d5", "c2c4", "e7e6"])

      # If 2.Nf3 -> ...Nf6
      add_line(["d2d4", "d7d5", "g1f3", "g8f6"])

      # If 2.Nc3 -> ...Nf6
      add_line(["d2d4", "d7d5", "b1c3", "g8f6"])

      # If they play something random after 1.d4, keep it solid: ...d5 then ...Nf6
      for white_second in ["e2e3", "g2g3", "f2f4", "b2b3", "c2c3", "a2a3"]:
        add_line(["d2d4", "d7d5", white_second, "g8f6"])

      return book

    self.opening_book = build_opening_book()


  def get_move(self, fen: str) -> Optional[str]:
    """
    Return a legal UCI move for the given FEN, or None if no legal moves exist.

    Move selection pipeline:
      0) Opening book
      1) Mate-in-1
      2) Avoid allowing opponent mate-in-1
      3) Beam alpha–beta (depth 4 by default), with LM move ordering
      4) Small heuristic move bonus at root (development + capture)
    """

    # ----------------------------
    # 0) Opening book (exact FEN lookup)
    # ----------------------------
    book_move = self.opening_book.get(fen)
    if book_move is not None:
      board_book = chess.Board(fen)
      if chess.Move.from_uci(book_move) in board_book.legal_moves:
        return book_move

    board = chess.Board(fen)
    legal = list(board.legal_moves)
    if not legal:
      return None

    if len(legal) == 1:
      return legal[0].uci()

    # ----------------------------
    # 1) Immediate tactical win: mate in 1
    # ----------------------------
    mate_move = self._find_mate_in_1(board)
    if mate_move is not None:
      return mate_move

    try:
      # ----------------------------
      # 2) Tactical safety: avoid allowing opponent mate in 1
      # ----------------------------
      safe_moves = []
      for m in legal:
        if not self._allows_opponent_mate_in_1(board, m):
          safe_moves.append(m)
      candidates = safe_moves if safe_moves else legal

      # Search/beam settings (tune safely)
      # Depth: 3 when branching is manageable, else 2
      # (You can tune these thresholds)
      DEPTH = 4 if len(candidates) <= 22 else 3

      # Root caps
      ROOT_LM_POOL = min(len(candidates), 24)  # hard cap 18–24; we'll ensure >=18 below
      K_ROOT = min(12, ROOT_LM_POOL)           # how many root moves we actually search

      # Beam below root: 5–7
      K_MAX = 7
      K_MIN = 6

      LAMBDA = 0.25

      # ----------------------------
      # 3) Root LM ordering (ONLY at root), on a capped pool.
      # First cheaply pick a pool (checks/captures/promos etc.)
      # ----------------------------
      
      pool_size = max(18, ROOT_LM_POOL)  # ensure 18–24 behavior
      pool_size = min(pool_size, len(candidates))

      root_pool = self._top_k_heuristic(board, candidates, k=pool_size)

      root_uci = [m.uci() for m in root_pool]
      root_scores = self._score_legal_moves(fen, root_uci)

      top_root = self._top_k_by_score(root_uci, root_scores, k=K_ROOT)
      uci_to_move = {m.uci(): m for m in root_pool}

      # cand_pairs is now only top K_ROOT, ordered by LM score
      cand_pairs = [(uci_to_move[uci], sc) for (uci, sc) in top_root if uci in uci_to_move]

      # Decide whether we are maximizing or minimizing White-eval at the root
      root_is_white = (board.turn == chess.WHITE)

      # Initialize best value from White perspective
      best_val = -10**9 if root_is_white else 10**9
      best_move = cand_pairs[0][0]  # fallback to best LM move

      for m, my_lm_score in cand_pairs[: min(K_ROOT, len(cand_pairs))]:
        # Root move bonus (development + capture)
        bonus = self._move_bonus(board, m, fen)

        board.push(m)

        # If we just delivered mate, always take it
        if board.is_checkmate():
          board.pop()
          return m.uci()

        # Alpha-beta from the position after our move
        # maximizing_for_white should reflect whose turn it is now
        val = self._alphabeta_beam(
          board=board,
          depth=DEPTH - 1,
          alpha=-10**9,
          beta=10**9,
          k_max=K_MAX,
          k_min=K_MIN,
        )
        board.pop()

        # Root combines position eval + small move bonus
        val = val + int(30 * LAMBDA * bonus)

        if root_is_white:
          if val > best_val:
            best_val = val
            best_move = m
        else:
          if val < best_val:
            best_val = val
            best_move = m

      return best_move.uci()

    except Exception:
      return candidates[0].uci() if candidates else legal[0].uci()

  # ============================================================================
  # Tactical helpers
  # ============================================================================

  def _find_mate_in_1(self, board: chess.Board) -> Optional[str]:
    """Return a UCI move that gives immediate checkmate, if any exists."""
    for m in board.legal_moves:
      board.push(m)
      is_mate = board.is_checkmate()
      board.pop()
      if is_mate:
        return m.uci()
    return None

  def _allows_opponent_mate_in_1(self, board: chess.Board, move: chess.Move) -> bool:
    """Return True if, after playing `move`, the opponent has a mate-in-1 reply."""
    board.push(move)
    try:
      for reply in board.legal_moves:
        board.push(reply)
        mate = board.is_checkmate()
        board.pop()
        if mate:
          return True
      return False
    finally:
      board.pop()

  def _development_bonus(self, board: chess.Board, move: chess.Move, ply: int) -> float:
    """
    Small positional bonus to favor sensible development:
      - Develop knights/bishops early
      - Castle early
      - Avoid moving the same piece repeatedly in the opening
      - Prefer central pawn pushes early (e4/d4/c4) over flank pawn pushes
      - Avoid early queen moves
      - Prefer connecting rooks / rook to central file later
    `ply` is half-move number from the FEN (0-based is fine).
    """
    # Only apply strongly in the opening
    if ply > 20:
      return 0.0

    piece = board.piece_at(move.from_square)
    if piece is None:
      return 0.0

    bonus = 0.0

    # Identify side to move
    us = board.turn

    # --- Castle bonus ---
    if board.is_castling(move):
      bonus += 3.5  # big push toward safety
      return bonus  # castle is usually a whole-plan move

    # --- Piece development bonuses ---
    # Knights: from starting squares to natural squares
    if piece.piece_type == chess.KNIGHT:
      # starting squares: b1/g1 or b8/g8
      if (us == chess.WHITE and move.from_square in [chess.B1, chess.G1]) or \
        (us == chess.BLACK and move.from_square in [chess.B8, chess.G8]):
        bonus += 1.2
      # prefer toward center: c3/f3 (white) or c6/f6 (black)
      to = move.to_square
      if us == chess.WHITE and to in [chess.C3, chess.F3, chess.D2, chess.E2]:
        bonus += 0.5
      if us == chess.BLACK and to in [chess.C6, chess.F6, chess.D7, chess.E7]:
        bonus += 0.5

    # Bishops: from starting squares to active diagonals
    if piece.piece_type == chess.BISHOP:
      if (us == chess.WHITE and move.from_square in [chess.C1, chess.F1]) or \
        (us == chess.BLACK and move.from_square in [chess.C8, chess.F8]):
        bonus += 1.0
      # small bias for common developing squares
      if us == chess.WHITE and move.to_square in [chess.C4, chess.B5, chess.D3, chess.E2, chess.F4]:
        bonus += 0.4
      if us == chess.BLACK and move.to_square in [chess.C5, chess.B4, chess.D6, chess.E7, chess.F5]:
        bonus += 0.4

    # --- Penalize early queen moves ---
    if piece.piece_type == chess.QUEEN and ply <= 12:
      bonus -= 1.2

    # --- Penalize rook moves very early (unless forced) ---
    if piece.piece_type == chess.ROOK and ply <= 14:
      bonus -= 0.6

    # --- Pawn move shaping ---
    if piece.piece_type == chess.PAWN:
      # Prefer central pawn moves in the opening
      if us == chess.WHITE and move.from_square in [chess.D2, chess.E2, chess.C2]:
        bonus += 0.4
      if us == chess.BLACK and move.from_square in [chess.D7, chess.E7, chess.C7]:
        bonus += 0.4

      # Penalize flank pawns early (a/h) and random pawn shuffles
      if us == chess.WHITE and move.from_square in [chess.A2, chess.H2] and ply <= 10:
        bonus -= 0.6
      if us == chess.BLACK and move.from_square in [chess.A7, chess.H7] and ply <= 10:
        bonus -= 0.6

    # --- Discourage moving same piece repeatedly early ---
    # Very simple heuristic: if this move returns a piece to its original square, punish.
    # (You can improve by tracking history, but this is cheap.)
    if move.to_square == move.from_square:
      bonus -= 0.5

    # --- Small bonus for giving check (often tactical), but keep it small ---
    if board.gives_check(move):
      bonus += 0.2

    return bonus

  def _piece_value(self, piece_type: int) -> int:
    return {
      chess.PAWN: 1,
      chess.KNIGHT: 3,
      chess.BISHOP: 3,
      chess.ROOK: 5,
      chess.QUEEN: 9,
      chess.KING: 0,
    }.get(piece_type, 0)


  def _capture_bonus(self, board: chess.Board, move: chess.Move, ply: int) -> float:
    """
    Small bonus to encourage making progress via captures.
    Tries to reward *good* captures more than random pawn grabs.
    """
    if not board.is_capture(move):
      return 0.0

    # De-emphasize capture greed in very late game
    if ply > 80:
      scale = 0.4
    else:
      scale = 1.0

    victim = board.piece_at(move.to_square)
    attacker = board.piece_at(move.from_square)

    # En passant: victim isn't on the to_square, so handle that.
    if victim is None and board.is_en_passant(move):
      victim_value = 1
    else:
      victim_value = self._piece_value(victim.piece_type) if victim else 1

    attacker_value = self._piece_value(attacker.piece_type) if attacker else 1

    # Prefer winning bigger pieces; slightly prefer captures by smaller attackers.
    # Keep these numbers small: this is just a tie-breaker.
    bonus = 0.25 * victim_value - 0.05 * attacker_value

    # If it's a capture that also gives check, add a tiny extra (often tactical).
    if board.gives_check(move):
      bonus += 0.10

    return scale * bonus

  def _ply_from_fen(self, fen: str) -> int:
    # FEN has halfmove_clock and fullmove_number; ply approx:
    # ply = (fullmove-1)*2 + (0 if white to move else 1)
    parts = fen.split()
    fullmove = int(parts[5])
    stm = parts[1]  # 'w' or 'b'
    return (fullmove - 1) * 2 + (0 if stm == "w" else 1)

  def _move_bonus(self, board: chess.Board, move: chess.Move, fen: str) -> float:
    """
    Combined heuristic bonus used to nudge move choice.
    """
    ply = self._ply_from_fen(fen)
    dev = self._development_bonus(board, move, ply)
    cap = self._capture_bonus(board, move, ply)
    return dev + cap

  # ============================================================================
  # Static evaluation (board scoring) for alpha-beta
  # Scores are from White perspective: positive = good for White.
  # ============================================================================

  def _material_eval_white(self, board: chess.Board) -> int:
    values = {
      chess.PAWN: 100,
      chess.KNIGHT: 320,
      chess.BISHOP: 330,
      chess.ROOK: 500,
      chess.QUEEN: 900,
      chess.KING: 0,
    }
    score = 0
    for pt, v in values.items():
      score += len(board.pieces(pt, chess.WHITE)) * v
      score -= len(board.pieces(pt, chess.BLACK)) * v
    return score

  def _pst_eval_white(self, board: chess.Board) -> int:
    """
    Tiny piece-square tables (only knights & bishops) to reduce shuffling
    and encourage centralization.
    """
    KNIGHT = [
      -50,-40,-30,-30,-30,-30,-40,-50,
      -40,-20,  0,  0,  0,  0,-20,-40,
      -30,  0, 10, 15, 15, 10,  0,-30,
      -30,  5, 15, 20, 20, 15,  5,-30,
      -30,  0, 15, 20, 20, 15,  0,-30,
      -30,  5, 10, 15, 15, 10,  5,-30,
      -40,-20,  0,  5,  5,  0,-20,-40,
      -50,-40,-30,-30,-30,-30,-40,-50,
    ]
    BISHOP = [
      -20,-10,-10,-10,-10,-10,-10,-20,
      -10,  0,  0,  0,  0,  0,  0,-10,
      -10,  0,  5, 10, 10,  5,  0,-10,
      -10,  5,  5, 10, 10,  5,  5,-10,
      -10,  0, 10, 10, 10, 10,  0,-10,
      -10, 10, 10, 10, 10, 10, 10,-10,
      -10,  5,  0,  0,  0,  0,  5,-10,
      -20,-10,-10,-10,-10,-10,-10,-20,
    ]

    def idx(sq: int, color: bool) -> int:
      return sq if color == chess.WHITE else chess.square_mirror(sq)

    score = 0
    for sq in board.pieces(chess.KNIGHT, chess.WHITE):
      score += KNIGHT[idx(sq, chess.WHITE)]
    for sq in board.pieces(chess.KNIGHT, chess.BLACK):
      score -= KNIGHT[idx(sq, chess.BLACK)]

    for sq in board.pieces(chess.BISHOP, chess.WHITE):
      score += BISHOP[idx(sq, chess.WHITE)]
    for sq in board.pieces(chess.BISHOP, chess.BLACK):
      score -= BISHOP[idx(sq, chess.BLACK)]

    return score

  def _mobility_eval_white(self, board: chess.Board) -> int:
    """
    Mobility: (#legal moves for White) - (#legal moves for Black), scaled.
    We compute it by temporarily switching the side to move using a null move.
    """
    try:
      stm = board.turn
      my_moves = board.legal_moves.count()

      board.push(chess.Move.null())
      opp_moves = board.legal_moves.count()
      board.pop()

      # If side-to-move was White, my_moves are White's mobility; else Black's.
      if stm == chess.WHITE:
        return 2 * (my_moves - opp_moves)
      else:
        return -2 * (my_moves - opp_moves)
    except Exception:
      return 0

  def _static_eval_white(self, board: chess.Board) -> int:
    """
    Terminal-aware static evaluation from White perspective.
    """
    if board.is_checkmate():
      # side to move is checkmated
      return -999999 if board.turn == chess.WHITE else 999999

    if board.is_stalemate() or board.is_insufficient_material():
      return 0

    return (
      self._material_eval_white(board)
      + self._pst_eval_white(board)
      + self._mobility_eval_white(board)
    )

  # ============================================================================
  # LM-based move ordering for alpha-beta (beam search)
  # ============================================================================

  def _ordered_moves_lm(
  self,
  board: chess.Board,
  k: int,
  prefilter_mult: int = 4,
  prefilter_min: int = 18,
  ) -> List[chess.Move]:
    """
    Return up to k legal moves ordered by LM score descending.

    Uses a cheap heuristic prefilter first to reduce LM scoring cost:
      - score only top M moves by (check/capture/promo), where
        M = max(prefilter_min, min(len(legal), k * prefilter_mult))
    """
    legal = list(board.legal_moves)
    if not legal:
      return []

    # Heuristic prefilter: only score a subset with the LM
    M = max(prefilter_min, min(len(legal), k * prefilter_mult))
    pool = self._top_k_heuristic(board, legal, k=M)

    moves_uci = [m.uci() for m in pool]
    scores = self._score_legal_moves(board.fen(), moves_uci)

    # Use helper to get top-k (uci, score) pairs
    top = self._top_k_by_score(moves_uci, scores, k=k)
    top_set = {uci for (uci, _) in top}

    # Preserve the LM rank order from `top`
    uci_to_move = {m.uci(): m for m in pool}
    return [uci_to_move[uci] for (uci, _) in top if uci in uci_to_move and uci in top_set]

  # ============================================================================
  # Beam alpha-beta
  # ============================================================================

  def _alphabeta_beam(
  self,
  board: chess.Board,
  depth: int,
  alpha: int,
  beta: int,
  k_max: int,
  k_min: int,
  ) -> int:
    if depth <= 0 or board.is_game_over(claim_draw=False):
      return self._static_eval_white(board)

    white_to_move = (board.turn == chess.WHITE)
    k = k_max if white_to_move else k_min

    # Heuristic ordering only (fast)
    moves = self._ordered_moves_heuristic(board, k=k)
    if not moves:
      return self._static_eval_white(board)

    if white_to_move:
      value = -10**9
      for m in moves:
        board.push(m)
        value = max(value, self._alphabeta_beam(board, depth - 1, alpha, beta, k_max, k_min))
        board.pop()
        alpha = max(alpha, value)
        if alpha >= beta:
          break
      return value
    else:
      value = 10**9
      for m in moves:
        board.push(m)
        value = min(value, self._alphabeta_beam(board, depth - 1, alpha, beta, k_max, k_min))
        board.pop()
        beta = min(beta, value)
        if alpha >= beta:
          break
      return value

  # ============================================================================
  # Candidate selection helpers
  # ============================================================================

  def _top_k_heuristic(self, board: chess.Board, moves: List[chess.Move], k: int = 12) -> List[chess.Move]:
    """
    Optional: a cheap heuristic filter you can use to reduce scoring cost.
    Prioritizes checks, captures, and promotions.
    """
    def score(m: chess.Move) -> int:
      s = 0
      if board.gives_check(m):
        s += 1000
      if board.is_capture(m):
        s += 500
      if m.promotion is not None:
        s += 1000
      return s

    moves_sorted = sorted(moves, key=score, reverse=True)
    return moves_sorted[: min(k, len(moves_sorted))]

  def _top_k_by_score(self, moves_uci: List[str], scores: List[float], k: int) -> List[Tuple[str, float]]:
    """Return the top-k (move, score) pairs sorted by score descending."""
    pairs = list(zip(moves_uci, scores))
    pairs.sort(key=lambda x: x[1], reverse=True)
    return pairs[: min(k, len(pairs))]

  def _ordered_moves_heuristic(self, board: chess.Board, k: int) -> List[chess.Move]:
    """
    Cheap move ordering for search nodes (below root).
    Prioritize: checks, promotions, captures (MVV-ish), then a tiny development bias.
    """
    ply = self._ply_from_fen(board.fen())

    def score(m: chess.Move) -> float:
      s = 0.0

      if m.promotion is not None:
        s += 10000.0

      if board.gives_check(m):
        s += 2000.0

      if board.is_capture(m):
        victim = board.piece_at(m.to_square)
        attacker = board.piece_at(m.from_square)

        # en-passant victim not on to_square
        victim_val = 1 if (victim is None and board.is_en_passant(m)) else (self._piece_value(victim.piece_type) if victim else 1)
        attacker_val = self._piece_value(attacker.piece_type) if attacker else 1

        # prefer winning larger pieces, slightly prefer capturing with smaller pieces
        s += 500.0 * victim_val - 50.0 * attacker_val

      # small opening bias (won't dominate tactics)
      s += 30.0 * self._development_bonus(board, m, ply)

      return s

    moves = list(board.legal_moves)
    moves.sort(key=score, reverse=True)
    return moves[: min(k, len(moves))]

  # ============================================================================
  # LM scoring + caching
  # ============================================================================

  def _make_prompt(self, fen: str) -> str:
    """Prompt format: keep short and consistent; UCI moves only."""
    return f"You are a strong chess engine. Best move given FEN: {fen}\nMOVE:"

  @lru_cache(maxsize=4096)
  def _cached_scores(self, fen: str, moves_tuple: Tuple[str, ...]) -> Tuple[float, ...]:
    """
    Memoize scores for a given (fen, sorted_moves_tuple).
    This is useful because the same position can be evaluated multiple times.
    """
    return tuple(self._score_moves_batch(self._make_prompt(fen), list(moves_tuple)))

  def _score_legal_moves(self, fen: str, moves_uci: List[str]) -> List[float]:
    """
    Score each move in `moves_uci` as average log-probability under the LM:
      score(move) = avg_logP(move_tokens | prompt(FEN)).
    """
    moves_sorted = tuple(sorted(moves_uci))
    scores_sorted = self._cached_scores(fen, moves_sorted)

    # Map back to the original order
    score_map = {m: s for m, s in zip(moves_sorted, scores_sorted)}
    return [score_map[m] for m in moves_uci]

  # ----------------------------
  # KV-cache encoding helpers
  # ----------------------------

  def _encode_prompt(self, prompt: str) -> torch.Tensor:
    """Tokenize the prompt (no special tokens)."""
    ids = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]
    return ids.to(self.dev)

  def _encode_move(self, mv: str) -> torch.Tensor:
    """
    Tokenize a move string. Leading space improves tokenization consistency for GPT2-family.
    """
    ids = self.tokenizer(" " + mv, add_special_tokens=False, return_tensors="pt")["input_ids"]
    return ids.to(self.dev)


  @torch.inference_mode()
  def _score_moves_batch(self, prompt: str, moves: List[str]) -> List[float]:
    if not moves:
      return []

    # 1) Prompt forward pass once
    prompt_ids = self._encode_prompt(prompt)  # (1, Lp)
    out = self.model(input_ids=prompt_ids, use_cache=True)

    past = out.past_key_values                 # Cache object OR legacy tuple
    next_logits = out.logits[:, -1, :]         # (1, V)

    # 2) Encode all moves -> padded batch
    move_ids_list = [self._encode_move(mv).squeeze(0) for mv in moves]
    lens = [int(t.numel()) for t in move_ids_list]
    max_len = max(lens)

    B = len(moves)
    pad_id = self.tokenizer.pad_token_id

    move_ids = torch.full((B, max_len), fill_value=pad_id, device=self.dev, dtype=torch.long)
    move_attn = torch.zeros((B, max_len), device=self.dev, dtype=torch.bool)

    for i, t in enumerate(move_ids_list):
      n = int(t.numel())
      move_ids[i, :n] = t
      move_attn[i, :n] = True

    # 3) Expand cache to batch size B (cache-type aware)
    if hasattr(past, "batch_repeat_interleave"):
      # New Cache API (fast + correct)
      past_b = past.batch_repeat_interleave(B)
    elif isinstance(past, tuple):
      # Legacy tuple cache (k,v per layer) -> manual expand
      past_b = []
      for layer in past:
        k = layer[0]
        v = layer[1]
        past_b.append((
          k.expand(B, -1, -1, -1).contiguous(),
          v.expand(B, -1, -1, -1).contiguous(),
        ))
      past_b = tuple(past_b)
    else:
      # Last-resort: try a simple repeat if available
      if hasattr(past, "repeat_interleave"):
        past_b = past.repeat_interleave(B)
      else:
        raise TypeError(f"Unknown cache type: {type(past)}; cannot batch-expand")

    # 4) Batched token logprob accumulation
    token_logp_sum = torch.zeros((B,), device=self.dev)
    token_count = torch.zeros((B,), device=self.dev)

    # token 0 from prompt logits
    logits0 = next_logits.expand(B, -1)
    ids0 = move_ids[:, 0]
    mask0 = move_attn[:, 0]

    lp0 = torch.log_softmax(logits0, dim=-1).gather(1, ids0.unsqueeze(1)).squeeze(1)
    token_logp_sum += torch.where(mask0, lp0, torch.zeros_like(lp0))
    token_count += mask0.to(token_count.dtype)

    # advance cache with token 0
    cur_ids = ids0.unsqueeze(1)  # (B,1)
    out = self.model(input_ids=cur_ids, past_key_values=past_b, use_cache=True)
    past_b = out.past_key_values
    cur_logits = out.logits[:, -1, :]

    # tokens 1..max_len-1
    for t in range(1, max_len):
      ids_t = move_ids[:, t]
      mask_t = move_attn[:, t]

      lp_t = torch.log_softmax(cur_logits, dim=-1).gather(1, ids_t.unsqueeze(1)).squeeze(1)
      token_logp_sum += torch.where(mask_t, lp_t, torch.zeros_like(lp_t))
      token_count += mask_t.to(token_count.dtype)

      cur_ids = ids_t.unsqueeze(1)
      out = self.model(input_ids=cur_ids, past_key_values=past_b, use_cache=True)
      past_b = out.past_key_values
      cur_logits = out.logits[:, -1, :]

    return (token_logp_sum / token_count.clamp_min(1.0)).detach().cpu().tolist()
