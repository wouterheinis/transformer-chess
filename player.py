from __future__ import annotations

import random
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
    # Keep this small and reliable; always verify legality at runtime.
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
    """

    # ----------------------------
    # 0) Opening book (exact FEN lookup)
    # ----------------------------
    book_move = self.opening_book.get(fen)
    if book_move is not None:
      board_book = chess.Board(fen)
      if chess.Move.from_uci(book_move) in board_book.legal_moves:
        return book_move

    # ----------------------------
    # 1) Parse board and list legal moves
    # ----------------------------
    board = chess.Board(fen)
    legal = list(board.legal_moves)
    if not legal:
      return None

    if len(legal) == 1:
      return legal[0].uci()

    # ----------------------------
    # 2) Immediate tactical win: mate in 1
    # ----------------------------
    mate_move = self._find_mate_in_1(board)
    if mate_move is not None:
      return mate_move

    try:
      # ----------------------------
      # 3) Tactical safety: avoid allowing opponent mate in 1
      # ----------------------------
      safe_moves = []
      for m in legal:
        if not self._allows_opponent_mate_in_1(board, m):
          safe_moves.append(m)

      candidates = safe_moves if safe_moves else legal

      # ----------------------------
      # 4) Score candidate moves with the LM (legal-move rescoring)
      # ----------------------------
      moves_uci = [m.uci() for m in candidates]
      scores = self._score_legal_moves(fen, moves_uci)

      # Consider only top-K moves under the model, then do a cheap 2-ply lookahead.
      top = self._top_k_by_score(moves_uci, scores, k=5)

      # ----------------------------
      # 5) 2-ply "minimax-lite": maximize (my_score - opponent_best_reply_score)
      # ----------------------------
      best_move = top[0][0]
      best_value = float("-inf")

      for mv, my_score in top:
        m = chess.Move.from_uci(mv)

        bonus = self._move_bonus(board, m, fen)

        board.push(m)

        # If this move gives immediate checkmate, take it.
        if board.is_checkmate():
          board.pop()
          return mv

        # --- opponent reply score ---
        if self._is_forcing_move(board, m):
          # We are already in the position after pushing our move (board.push(m) happened above).
          # Use a deeper reply model: opponent reply + our best response.
          opp_score = self._best_reply_score_with_forcing_extension(board, k_opp=4, k_us2=7)
        else:
          # Cheap 2-ply behavior for quiet moves
          opp_score = self._best_reply_score(board)
        board.pop()


        LAMBDA = 0.25  # start small; tune 0.15..0.40
        value = (my_score - opp_score) + LAMBDA * bonus
        if value > best_value:
          best_value = value
          best_move = mv

      return best_move

    except Exception:
      # Never return illegal moves or crash; fall back to a random legal move.
      return random.choice(legal).uci()

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

  def _best_reply_score(self, board: chess.Board) -> float:
    """
    Return the score of the opponent's best move (according to the LM) in this position.
    If no legal moves exist, return 0.0.
    """
    legal = list(board.legal_moves)
    if not legal:
      return 0.0

    moves_uci = [m.uci() for m in legal]
    scores = self._score_legal_moves(board.fen(), moves_uci)
    return max(scores)

  def _is_forcing_move(self, board: chess.Board, move: chess.Move) -> bool:
    """
    Return True if this move is tactically forcing enough to justify a deeper look.
    We consider: checks, captures, promotions.
    """
    if move.promotion is not None:
      return True
    if board.is_capture(move):
      return True
    if board.gives_check(move):
      return True
    return False


  def _best_reply_score_with_forcing_extension(
      self,
      board_after_our_move: chess.Board,
      k_opp: int = 3,
      k_us2: int = 3
    ) -> float:
    """
    Opponent best-reply score with one extra ply (beam minimax) from the position
    after we have already played our candidate move.

    Returns a number to subtract from our candidate value.
    Larger = worse for us (opponent has a strong reply).
    """

    # If opponent has no moves, game is over; no "reply" penalty.
    opp_legal = list(board_after_our_move.legal_moves)
    if not opp_legal:
      return 0.0

    fen_opp = board_after_our_move.fen()
    opp_moves_uci = [m.uci() for m in opp_legal]
    opp_scores = self._score_legal_moves(fen_opp, opp_moves_uci)
    opp_top = self._top_k_by_score(opp_moves_uci, opp_scores, k=k_opp)

    # Opponent tries to MINIMIZE our future (i.e., pick the reply that hurts us most).
    worst_for_us = float("inf")

    for reply_uci, _reply_score in opp_top:
      reply = chess.Move.from_uci(reply_uci)
      board_after_our_move.push(reply)

      # If opponent just checkmated us, this reply is crushing.
      if board_after_our_move.is_checkmate():
        board_after_our_move.pop()
        return 999.0

      # Our best response score after their reply (max node)
      us2_legal = list(board_after_our_move.legal_moves)
      if not us2_legal:
        us_best = 0.0
      else:
        fen_us2 = board_after_our_move.fen()
        us2_moves_uci = [m.uci() for m in us2_legal]
        us2_scores = self._score_legal_moves(fen_us2, us2_moves_uci)
        us2_top = self._top_k_by_score(us2_moves_uci, us2_scores, k=k_us2)
        us_best = us2_top[0][1] if us2_top else 0.0

      board_after_our_move.pop()

      # Opponent chooses the reply that leaves us with the lowest "best response"
      if us_best < worst_for_us:
        worst_for_us = us_best

    # If something weird happened, fall back safely.
    if worst_for_us == float("inf"):
      return max(opp_scores)

    return worst_for_us

  # ============================================================================
  # LM scoring + caching
  # ============================================================================

  def _make_prompt(self, fen: str) -> str:
    """Prompt format: keep short and consistent; UCI moves only."""
    return f"FEN: {fen}\nMOVE:"

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
    """
    Fast scoring using KV cache:
      - Run the prompt once to get `past_key_values`.
      - Score each candidate move by feeding only its move tokens step-by-step.
      - Return length-normalized log-prob sums (average logprob per token).
    """
    # 1) Run prompt once -> KV cache
    prompt_ids = self._encode_prompt(prompt)  # (1, Lp)
    out = self.model(input_ids=prompt_ids, use_cache=True)
    past = out.past_key_values

    # Logits for the first token after the prompt
    next_logits = out.logits[:, -1, :]  # (1, V)

    # 2) Encode all moves (variable-length token sequences)
    move_ids_list = [self._encode_move(mv).squeeze(0) for mv in moves]  # each (Lm,)
    lens = [int(t.numel()) for t in move_ids_list]
    max_len = max(lens)

    B = len(moves)
    pad_id = self.tokenizer.pad_token_id

    # Padded move token matrix
    move_ids = torch.full((B, max_len), fill_value=pad_id, device=self.dev, dtype=torch.long)
    move_attn = torch.zeros((B, max_len), device=self.dev, dtype=torch.bool)

    for i, t in enumerate(move_ids_list):
      move_ids[i, : t.numel()] = t
      move_attn[i, : t.numel()] = True

    # 3) Expand KV cache to batch size B
    past_b = []
    for (k, v) in past:
      past_b.append((
        k.expand(B, -1, -1, -1).contiguous(),
        v.expand(B, -1, -1, -1).contiguous()
      ))
    past_b = tuple(past_b)

    # 4) Accumulate token log-probs for each move
    token_logp_sum = torch.zeros((B,), device=self.dev)
    token_count = torch.zeros((B,), device=self.dev)

    # Token 0 is predicted from the prompt logits
    logits0 = next_logits.expand(B, -1)
    ids0 = move_ids[:, 0]
    mask0 = move_attn[:, 0]

    lp0 = torch.log_softmax(logits0, dim=-1).gather(1, ids0.unsqueeze(1)).squeeze(1)
    token_logp_sum += torch.where(mask0, lp0, torch.zeros_like(lp0))
    token_count += mask0.to(token_count.dtype)

    # Advance cache with token 0 (pads are fine; they're masked out)
    cur_ids = ids0.unsqueeze(1)  # (B, 1)
    out = self.model(input_ids=cur_ids, past_key_values=past_b, use_cache=True)
    past_b = out.past_key_values
    cur_logits = out.logits[:, -1, :]  # predicts token 1

    # Tokens 1..max_len-1
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

    # 5) Length-normalize to reduce bias due to different tokenization lengths
    scores = (token_logp_sum / token_count.clamp_min(1.0)).detach().cpu().tolist()
    return scores
