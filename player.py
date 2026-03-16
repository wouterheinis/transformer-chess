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

# ---------------------------------------------------------------------------
# PyTorch performance flags
# ---------------------------------------------------------------------------
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")


def _device() -> torch.device:
    """Return CUDA if available, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Search / beam constants
# ---------------------------------------------------------------------------
DEPTH_MANY_CANDIDATES = 3   # Search depth when there are many root candidates
DEPTH_FEW_CANDIDATES  = 4   # Search depth when candidates are manageable (≤22)
CANDIDATE_THRESHOLD   = 22  # Root candidate count below which we use the deeper depth

ROOT_LM_POOL_SIZE = 24  # How many moves to score with the LM at the root
K_ROOT            = 12  # How many top-LM moves to actually search at the root

K_MAX = 7   # Beam width for the side to move inside the tree
K_MIN = 6   # Beam width for the opponent's side inside the tree

LM_BONUS_LAMBDA = 0.25  # Weight of the LM bonus added to the alpha-beta score


# ===========================================================================
# TransformerPlayer
# ===========================================================================

class TransformerPlayer(Player):
    """
    Chess engine that combines a pre-trained transformer (DistilGPT-2) with
    classical alpha-beta search.

    Architecture
    ------------
    1. Opening book  – a handcrafted set of lines for the first few moves.
    2. LM policy     – DistilGPT-2 scores UCI moves conditioned on the FEN.
    3. Alpha-beta    – beam-limited search guided by heuristic move ordering.

    The transformer acts as a *policy* at the root: it ranks candidates so
    that only the most plausible moves are explored by the alpha-beta search.
    No fine-tuning was performed; the model is used entirely out-of-the-box.
    """

    # -----------------------------------------------------------------------
    # Initialisation
    # -----------------------------------------------------------------------

    def __init__(self, name: str):
        super().__init__(name)
        self._init_model()
        self.opening_book = self._build_opening_book()

    def _init_model(self) -> None:
        """Load DistilGPT-2 and move it to the best available device."""
        self.model_name = "distilgpt2"
        self.tokenizer  = AutoTokenizer.from_pretrained(self.model_name)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        self.model.eval()

        self.dev = _device()
        self.model.to(self.dev)

        # Use half-precision on GPU to reduce memory and speed up inference
        if self.dev.type == "cuda":
            self.model.half()

    # -----------------------------------------------------------------------
    # Opening book construction
    # -----------------------------------------------------------------------

    @staticmethod
    def _build_opening_book() -> Dict[str, str]:
        """
        Build a FEN → UCI move mapping from a set of hard-coded opening lines.

        For every position in each line where it is *our* turn, we record the
        next move as the book response.  If two lines share a FEN, the first
        one wins (``setdefault``).
        """
        book: Dict[str, str] = {}

        def add_line(moves: List[str]) -> None:
            """Walk a sequence of UCI moves from the starting position and
            store each FEN → next-move pair."""
            board = chess.Board()
            for uci in moves:
                mv = chess.Move.from_uci(uci)
                if mv not in board.legal_moves:
                    break                       # Stop if any move is illegal
                book.setdefault(board.fen(), uci)
                board.push(mv)

        # -------------------------------------------------------------------
        # White repertoire – 1.e4
        # -------------------------------------------------------------------
        add_line(["e2e4"])

        # vs 1…e5: Italian / Giuoco Piano
        add_line(["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "d2d3"])
        # vs 1…e5: Scotch-style centre break
        add_line(["e2e4", "e7e5", "g1f3", "b8c6", "d2d4", "e5d4", "f3d4"])

        # vs 1…c5 Sicilian – main Nc3 plans
        add_line(["e2e4", "c7c5", "g1f3", "d7d6", "d2d4"])
        add_line(["e2e4", "c7c5", "g1f3", "e7e6", "d2d4"])
        add_line(["e2e4", "c7c5", "g1f3", "g8f6", "d2d4"])

        # vs 1…e6 French
        add_line(["e2e4", "e7e6", "d2d4", "d7d5", "b1c3"])

        # vs 1…c6 Caro-Kann
        add_line(["e2e4", "c7c6", "d2d4", "d7d5", "b1c3"])

        # vs 1…d5 Scandinavian
        add_line(["e2e4", "d7d5", "e4d5", "d8d5", "b1c3"])

        # vs various early flank / unusual replies – just develop
        for black_reply in ["a7a6", "h7h6", "g7g6", "b7b6", "f7f6",
                            "g7g5", "d7d6", "b8c6", "g8f6"]:
            add_line(["e2e4", black_reply, "g1f3"])

        # vs 1…g6 – claim the centre
        add_line(["e2e4", "g7g6", "d2d4", "f8g7", "f1c4"])

        # -------------------------------------------------------------------
        # Black repertoire vs 1.e4 – Caro-Kann (1…c6)
        # -------------------------------------------------------------------
        add_line(["e2e4", "c7c6"])

        # Main line: 2.d4 d5 3.Nc3 dxe4 (Classical)
        add_line(["e2e4", "c7c6", "d2d4", "d7d5", "b1c3", "d5e4"])
        add_line(["e2e4", "c7c6", "g1f3", "d7d5"])
        add_line(["e2e4", "c7c6", "b1c3", "d7d5"])

        # vs White's various second moves – always answer …d5
        for white_second in ["f1c4", "f2f4", "h2h3", "a2a3", "g2g3", "b2b3"]:
            add_line(["e2e4", "c7c6", white_second, "d7d5"])

        # -------------------------------------------------------------------
        # Black repertoire vs 1.d4 – Queen's Gambit Declined structure
        # -------------------------------------------------------------------
        add_line(["d2d4", "d7d5"])

        # vs 2.c4 – QGD
        add_line(["d2d4", "d7d5", "c2c4", "e7e6"])

        # vs 2.Nf3 / 2.Nc3 – develop knight
        add_line(["d2d4", "d7d5", "g1f3", "g8f6"])
        add_line(["d2d4", "d7d5", "b1c3", "g8f6"])

        # vs various quiet second moves – develop knight
        for white_second in ["e2e3", "g2g3", "f2f4", "b2b3", "c2c3", "a2a3"]:
            add_line(["d2d4", "d7d5", white_second, "g8f6"])

        return book

    # -----------------------------------------------------------------------
    # Main entry point
    # -----------------------------------------------------------------------

    def get_move(self, fen: str) -> Optional[str]:
        """
        Choose a move for the position encoded by *fen*.

        Decision pipeline
        -----------------
        0. Opening book – return immediately if position is covered.
        1. Forced moves – return immediately if only one legal move.
        2. Mate-in-1    – return immediately if a mating move exists.
        3. Safety filter – discard moves that allow opponent mate-in-1.
        4. LM policy    – rank the best candidates with DistilGPT-2.
        5. Alpha-beta   – search each candidate and pick the best score.
        """

        # 0) Opening book
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

        # 1) Instant win: mate in 1
        mate_move = self._find_mate_in_1(board)
        if mate_move is not None:
            return mate_move

        try:
            # 2) Safety filter: avoid moves that gift the opponent mate-in-1
            safe_moves = [m for m in legal if not self._allows_opponent_mate_in_1(board, m)]
            candidates = safe_moves if safe_moves else legal

            # 3) Search depth depends on how many candidates remain
            depth = DEPTH_FEW_CANDIDATES if len(candidates) <= CANDIDATE_THRESHOLD \
                    else DEPTH_MANY_CANDIDATES

            # 4) LM policy at the root
            #    a) Pre-filter with cheap heuristics → root_pool
            pool_size = min(max(18, ROOT_LM_POOL_SIZE), len(candidates))
            root_pool = self._top_k_heuristic(board, candidates, k=pool_size)

            #    b) Score the pool with the LM
            root_uci    = [m.uci() for m in root_pool]
            root_scores = self._score_legal_moves(fen, root_uci)

            #    c) Keep the top-K_ROOT moves ranked by LM score
            top_root     = self._top_k_by_score(root_uci, root_scores, k=K_ROOT)
            uci_to_move  = {m.uci(): m for m in root_pool}
            cand_pairs   = [(uci_to_move[uci], sc)
                            for (uci, sc) in top_root if uci in uci_to_move]

            # 5) Alpha-beta over the LM-ranked candidates
            root_is_white = (board.turn == chess.WHITE)
            best_val  = -10**9 if root_is_white else 10**9
            best_move = cand_pairs[0][0]  # Fallback: best LM move

            for m, _ in cand_pairs[:K_ROOT]:
                bonus = self._move_bonus(board, m, fen)
                board.push(m)

                # Immediate checkmate – no need to search further
                if board.is_checkmate():
                    board.pop()
                    return m.uci()

                val = self._alphabeta_beam(
                    board=board,
                    depth=depth - 1,
                    alpha=-10**9,
                    beta=10**9,
                    k_max=K_MAX,
                    k_min=K_MIN,
                )
                board.pop()

                # Blend alpha-beta score with small opening-heuristic bonus
                val += int(30 * LM_BONUS_LAMBDA * bonus)

                if root_is_white:
                    if val > best_val:
                        best_val, best_move = val, m
                else:
                    if val < best_val:
                        best_val, best_move = val, m

            return best_move.uci()

        except Exception:
            # If anything goes wrong, fall back to the first safe candidate
            return candidates[0].uci() if candidates else legal[0].uci()

    # -----------------------------------------------------------------------
    # Tactical helpers
    # -----------------------------------------------------------------------

    def _find_mate_in_1(self, board: chess.Board) -> Optional[str]:
        """Return a UCI move that delivers immediate checkmate, if one exists."""
        for m in board.legal_moves:
            board.push(m)
            is_mate = board.is_checkmate()
            board.pop()
            if is_mate:
                return m.uci()
        return None

    def _allows_opponent_mate_in_1(self, board: chess.Board, move: chess.Move) -> bool:
        """Return True if playing *move* leaves the opponent a mating reply."""
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

    # -----------------------------------------------------------------------
    # Opening / positional heuristics
    # -----------------------------------------------------------------------

    def _development_bonus(self, board: chess.Board, move: chess.Move, ply: int) -> float:
        """
        Return a small floating-point bonus that nudges the engine toward
        sound development principles in the opening (ply ≤ 20).

        Bonuses reward:
          - Castling
          - Moving knights and bishops off the back rank toward active squares
          - Central pawn advances

        Penalties discourage:
          - Early queen and rook moves
          - Flank pawn shuffles in the first 10 moves
        """
        if ply > 20:
            return 0.0

        piece = board.piece_at(move.from_square)
        if piece is None:
            return 0.0

        bonus = 0.0
        us    = board.turn

        # Castling is always good in the opening
        if board.is_castling(move):
            return 3.5

        if piece.piece_type == chess.KNIGHT:
            # Reward moving off the starting square …
            starting = [chess.B1, chess.G1] if us == chess.WHITE else [chess.B8, chess.G8]
            if move.from_square in starting:
                bonus += 1.2
            # … and landing on a central development square
            central_white = [chess.C3, chess.F3, chess.D2, chess.E2]
            central_black = [chess.C6, chess.F6, chess.D7, chess.E7]
            if (us == chess.WHITE and move.to_square in central_white) or \
               (us == chess.BLACK and move.to_square in central_black):
                bonus += 0.5

        elif piece.piece_type == chess.BISHOP:
            # Reward moving off the back rank …
            starting = [chess.C1, chess.F1] if us == chess.WHITE else [chess.C8, chess.F8]
            if move.from_square in starting:
                bonus += 1.0
            # … and landing on an active diagonal
            active_white = [chess.C4, chess.B5, chess.D3, chess.E2, chess.F4]
            active_black = [chess.C5, chess.B4, chess.D6, chess.E7, chess.F5]
            if (us == chess.WHITE and move.to_square in active_white) or \
               (us == chess.BLACK and move.to_square in active_black):
                bonus += 0.4

        elif piece.piece_type == chess.QUEEN and ply <= 12:
            bonus -= 1.2   # Premature queen activation is risky

        elif piece.piece_type == chess.ROOK and ply <= 14:
            bonus -= 0.6   # Rooks belong on open files, not the opening

        elif piece.piece_type == chess.PAWN:
            # Reward central pawn moves
            central_pawns_white = [chess.D2, chess.E2, chess.C2]
            central_pawns_black = [chess.D7, chess.E7, chess.C7]
            if (us == chess.WHITE and move.from_square in central_pawns_white) or \
               (us == chess.BLACK and move.from_square in central_pawns_black):
                bonus += 0.4

            # Penalise time-wasting flank pawn moves early on
            flank_pawns_white = [chess.A2, chess.H2]
            flank_pawns_black = [chess.A7, chess.H7]
            if ply <= 10:
                if (us == chess.WHITE and move.from_square in flank_pawns_white) or \
                   (us == chess.BLACK and move.from_square in flank_pawns_black):
                    bonus -= 0.6

        # Small bonus for giving check (disrupts opponent's development)
        if board.gives_check(move):
            bonus += 0.2

        return bonus

    # Standard centipawn piece values used throughout
    _PIECE_VALUES = {
        chess.PAWN:   1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK:   5,
        chess.QUEEN:  9,
        chess.KING:   0,
    }

    def _piece_value(self, piece_type: int) -> int:
        return self._PIECE_VALUES.get(piece_type, 0)

    def _capture_bonus(self, board: chess.Board, move: chess.Move, ply: int) -> float:
        """
        Return a bonus that encourages capturing higher-value pieces.

        Capture greed is down-scaled in the endgame (ply > 80) to avoid
        unsound material grabbing when the position requires caution.
        """
        if not board.is_capture(move):
            return 0.0

        scale = 0.4 if ply > 80 else 1.0

        victim   = board.piece_at(move.to_square)
        attacker = board.piece_at(move.from_square)

        # En-passant: victim square is not the destination square
        if victim is None and board.is_en_passant(move):
            victim_value = 1
        else:
            victim_value = self._piece_value(victim.piece_type) if victim else 1

        attacker_value = self._piece_value(attacker.piece_type) if attacker else 1

        bonus = 0.25 * victim_value - 0.05 * attacker_value

        if board.gives_check(move):
            bonus += 0.10   # Capture with check is especially forcing

        return scale * bonus

    def _ply_from_fen(self, fen: str) -> int:
        """Convert the FEN fullmove counter to a half-move (ply) count."""
        parts     = fen.split()
        fullmove  = int(parts[5])
        side      = parts[1]   # 'w' or 'b'
        return (fullmove - 1) * 2 + (0 if side == "w" else 1)

    def _move_bonus(self, board: chess.Board, move: chess.Move, fen: str) -> float:
        """Combined heuristic bonus: development + capture value."""
        ply = self._ply_from_fen(fen)
        return self._development_bonus(board, move, ply) + \
               self._capture_bonus(board, move, ply)

    # -----------------------------------------------------------------------
    # Static evaluation for alpha-beta leaf nodes
    # -----------------------------------------------------------------------

    def _material_eval_white(self, board: chess.Board) -> int:
        """Material balance in centipawns from White's perspective."""
        values = {
            chess.PAWN:   100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK:   500,
            chess.QUEEN:  900,
            chess.KING:   0,
        }
        score = 0
        for pt, v in values.items():
            score += len(board.pieces(pt, chess.WHITE)) * v
            score -= len(board.pieces(pt, chess.BLACK)) * v
        return score

    def _pst_eval_white(self, board: chess.Board) -> int:
        """
        Piece-square table bonus for knights and bishops (from White's perspective).

        Rewards centralisation and discourages pieces lingering on the rim.
        Tables are indexed from White's perspective; Black pieces use the
        mirrored square index.
        """
        # fmt: off
        KNIGHT_PST = [
            -50,-40,-30,-30,-30,-30,-40,-50,
            -40,-20,  0,  0,  0,  0,-20,-40,
            -30,  0, 10, 15, 15, 10,  0,-30,
            -30,  5, 15, 20, 20, 15,  5,-30,
            -30,  0, 15, 20, 20, 15,  0,-30,
            -30,  5, 10, 15, 15, 10,  5,-30,
            -40,-20,  0,  5,  5,  0,-20,-40,
            -50,-40,-30,-30,-30,-30,-40,-50,
        ]
        BISHOP_PST = [
            -20,-10,-10,-10,-10,-10,-10,-20,
            -10,  0,  0,  0,  0,  0,  0,-10,
            -10,  0,  5, 10, 10,  5,  0,-10,
            -10,  5,  5, 10, 10,  5,  5,-10,
            -10,  0, 10, 10, 10, 10,  0,-10,
            -10, 10, 10, 10, 10, 10, 10,-10,
            -10,  5,  0,  0,  0,  0,  5,-10,
            -20,-10,-10,-10,-10,-10,-10,-20,
        ]
        # fmt: on

        def pst_index(sq: int, color: bool) -> int:
            """Return the PST index for *sq*, mirroring for Black."""
            return sq if color == chess.WHITE else chess.square_mirror(sq)

        score = 0
        for sq in board.pieces(chess.KNIGHT, chess.WHITE):
            score += KNIGHT_PST[pst_index(sq, chess.WHITE)]
        for sq in board.pieces(chess.KNIGHT, chess.BLACK):
            score -= KNIGHT_PST[pst_index(sq, chess.BLACK)]

        for sq in board.pieces(chess.BISHOP, chess.WHITE):
            score += BISHOP_PST[pst_index(sq, chess.WHITE)]
        for sq in board.pieces(chess.BISHOP, chess.BLACK):
            score -= BISHOP_PST[pst_index(sq, chess.BLACK)]

        return score

    def _mobility_eval_white(self, board: chess.Board) -> int:
        """
        Mobility bonus: (White legal moves) − (Black legal moves), scaled by 2.

        A null move is used to count the opponent's moves without altering the
        game state.
        """
        try:
            stm      = board.turn
            my_moves = board.legal_moves.count()

            board.push(chess.Move.null())
            opp_moves = board.legal_moves.count()
            board.pop()

            # Adjust sign depending on which side is to move
            if stm == chess.WHITE:
                return 2 * (my_moves - opp_moves)
            else:
                return -2 * (my_moves - opp_moves)
        except Exception:
            return 0

    def _static_eval_white(self, board: chess.Board) -> int:
        """
        Terminal-aware leaf evaluation from White's perspective.

        Returns ±999 999 for checkmate and 0 for draws.
        """
        if board.is_checkmate():
            return -999_999 if board.turn == chess.WHITE else 999_999

        if board.is_stalemate() or board.is_insufficient_material():
            return 0

        return (
            self._material_eval_white(board)
            + self._pst_eval_white(board)
            + self._mobility_eval_white(board)
        )

    # -----------------------------------------------------------------------
    # LM-based move ordering helpers
    # -----------------------------------------------------------------------

    def _ordered_moves_lm(
        self,
        board: chess.Board,
        k: int,
        prefilter_mult: int = 4,
        prefilter_min:  int = 18,
    ) -> List[chess.Move]:
        """
        Return up to *k* legal moves ranked by LM score (descending).

        A cheap heuristic pre-filter reduces the number of moves sent to the
        LM: we score only the top M moves by tactical priority, where
        M = max(prefilter_min, min(len(legal), k × prefilter_mult)).
        """
        legal = list(board.legal_moves)
        if not legal:
            return []

        M    = max(prefilter_min, min(len(legal), k * prefilter_mult))
        pool = self._top_k_heuristic(board, legal, k=M)

        moves_uci = [m.uci() for m in pool]
        scores    = self._score_legal_moves(board.fen(), moves_uci)

        top     = self._top_k_by_score(moves_uci, scores, k=k)
        top_set = {uci for (uci, _) in top}

        uci_to_move = {m.uci(): m for m in pool}
        return [uci_to_move[uci] for (uci, _) in top
                if uci in uci_to_move and uci in top_set]

    # -----------------------------------------------------------------------
    # Alpha-beta beam search
    # -----------------------------------------------------------------------

    def _alphabeta_beam(
        self,
        board: chess.Board,
        depth: int,
        alpha: int,
        beta:  int,
        k_max: int,
        k_min: int,
    ) -> int:
        """
        Beam-limited alpha-beta search returning a score from White's perspective.

        The beam restricts branching: k_max branches for the maximising side
        (White) and k_min for the minimising side (Black).  Move ordering uses
        cheap heuristics (captures, checks, promotions) rather than the LM to
        keep inner nodes fast.
        """
        if depth <= 0 or board.is_game_over(claim_draw=False):
            return self._static_eval_white(board)

        white_to_move = (board.turn == chess.WHITE)
        k = k_max if white_to_move else k_min

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
                    break   # Beta cut-off
            return value
        else:
            value = 10**9
            for m in moves:
                board.push(m)
                value = min(value, self._alphabeta_beam(board, depth - 1, alpha, beta, k_max, k_min))
                board.pop()
                beta = min(beta, value)
                if alpha >= beta:
                    break   # Alpha cut-off
            return value

    # -----------------------------------------------------------------------
    # Candidate selection helpers
    # -----------------------------------------------------------------------

    def _top_k_heuristic(
        self,
        board: chess.Board,
        moves: List[chess.Move],
        k: int = 12,
    ) -> List[chess.Move]:
        """
        Return the top-k moves ranked by a fast tactical heuristic.

        Priority: promotions = checks > captures > all others.
        Used as a cheap pre-filter before calling the (expensive) LM.
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

        return sorted(moves, key=score, reverse=True)[:k]

    def _top_k_by_score(
        self,
        moves_uci: List[str],
        scores:    List[float],
        k: int,
    ) -> List[Tuple[str, float]]:
        """Return the top-k (UCI move, score) pairs sorted by score descending."""
        pairs = sorted(zip(moves_uci, scores), key=lambda x: x[1], reverse=True)
        return pairs[:k]

    def _ordered_moves_heuristic(self, board: chess.Board, k: int) -> List[chess.Move]:
        """
        Order all legal moves by a fast heuristic for use *inside* the search tree.

        Priority (highest first):
          1. Promotions        (+10 000)
          2. Checks            (+2 000)
          3. Captures (MVV-LVA)
          4. Development bonus (small opening-only weight)
        """
        ply = self._ply_from_fen(board.fen())

        def score(m: chess.Move) -> float:
            s = 0.0

            if m.promotion is not None:
                s += 10_000.0

            if board.gives_check(m):
                s += 2_000.0

            if board.is_capture(m):
                victim   = board.piece_at(m.to_square)
                attacker = board.piece_at(m.from_square)

                # En-passant: captured pawn is not on the destination square
                victim_val   = 1 if (victim is None and board.is_en_passant(m)) \
                               else (self._piece_value(victim.piece_type) if victim else 1)
                attacker_val = self._piece_value(attacker.piece_type) if attacker else 1

                s += 500.0 * victim_val - 50.0 * attacker_val   # MVV-LVA

            s += 30.0 * self._development_bonus(board, m, ply)

            return s

        moves = list(board.legal_moves)
        moves.sort(key=score, reverse=True)
        return moves[:k]

    # -----------------------------------------------------------------------
    # LM scoring and caching
    # -----------------------------------------------------------------------

    def _make_prompt(self, fen: str) -> str:
        """Format the FEN string into the prompt the LM sees."""
        return f"FEN: {fen}\nMOVE:"

    @lru_cache(maxsize=4096)
    def _cached_scores(
        self,
        fen:         str,
        moves_tuple: Tuple[str, ...],
    ) -> Tuple[float, ...]:
        """
        Memoised wrapper around ``_score_moves_batch``.

        The moves are canonically sorted before caching so that different
        orderings of the same set always hit the same cache entry.
        """
        return tuple(self._score_moves_batch(self._make_prompt(fen), list(moves_tuple)))

    def _score_legal_moves(self, fen: str, moves_uci: List[str]) -> List[float]:
        """
        Return a per-move LM score: average log P(move tokens | FEN prompt).

        Internally sorts moves for cache efficiency then maps results back to
        the original order.
        """
        moves_sorted  = tuple(sorted(moves_uci))
        scores_sorted = self._cached_scores(fen, moves_sorted)

        score_map = dict(zip(moves_sorted, scores_sorted))
        return [score_map[m] for m in moves_uci]

    # -----------------------------------------------------------------------
    # KV-cache encoding helpers
    # -----------------------------------------------------------------------

    def _encode_prompt(self, prompt: str) -> torch.Tensor:
        """Tokenise *prompt* and return a (1, L) token-ID tensor on device."""
        ids = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]
        return ids.to(self.dev)

    def _encode_move(self, mv: str) -> torch.Tensor:
        """
        Tokenise a single UCI move string.

        A leading space is prepended so the tokeniser treats the move as a
        continuation (e.g. " e2e4") rather than a fresh sequence start.
        """
        ids = self.tokenizer(" " + mv, add_special_tokens=False, return_tensors="pt")["input_ids"]
        return ids.to(self.dev)

    @torch.inference_mode()
    def _score_moves_batch(self, prompt: str, moves: List[str]) -> List[float]:
        """
        Score all moves in a single batched forward pass through the LM.

        Algorithm
        ---------
        1. Run the prompt through the model *once* to obtain the KV cache and
           the logits for the first move token.
        2. Encode all moves into a padded (B × max_len) tensor.
        3. Expand the cached KV state to batch size B.
        4. Walk token-by-token across the move sequences, accumulating the
           log-probability of each real (non-padding) token.
        5. Return the *average* log-probability per move.
        """
        if not moves:
            return []

        # ── Step 1: Encode the prompt and run it once ──────────────────────
        prompt_ids  = self._encode_prompt(prompt)           # (1, L_prompt)
        out         = self.model(input_ids=prompt_ids, use_cache=True)
        past        = out.past_key_values
        next_logits = out.logits[:, -1, :]                  # (1, V)

        # ── Step 2: Pad all move sequences to the same length ──────────────
        move_ids_list = [self._encode_move(mv).squeeze(0) for mv in moves]
        lens    = [int(t.numel()) for t in move_ids_list]
        max_len = max(lens)
        B       = len(moves)
        pad_id  = self.tokenizer.pad_token_id

        move_ids  = torch.full((B, max_len), fill_value=pad_id,
                               device=self.dev, dtype=torch.long)
        move_attn = torch.zeros((B, max_len), device=self.dev, dtype=torch.bool)

        for i, t in enumerate(move_ids_list):
            n = int(t.numel())
            move_ids[i, :n]  = t
            move_attn[i, :n] = True

        # ── Step 3: Expand the KV cache to batch size B ────────────────────
        if hasattr(past, "batch_repeat_interleave"):
            # Transformers ≥ 4.x fast Cache API
            past_b = past.batch_repeat_interleave(B)
        elif isinstance(past, tuple):
            # Legacy tuple cache: (key, value) per layer
            past_b = tuple(
                (layer[0].expand(B, -1, -1, -1).contiguous(),
                 layer[1].expand(B, -1, -1, -1).contiguous())
                for layer in past
            )
        elif hasattr(past, "repeat_interleave"):
            past_b = past.repeat_interleave(B)
        else:
            raise TypeError(f"Unknown KV-cache type: {type(past)}")

        # ── Step 4: Accumulate per-token log-probabilities ─────────────────
        token_logp_sum = torch.zeros((B,), device=self.dev)
        token_count    = torch.zeros((B,), device=self.dev)

        # Token 0: scored against the prompt's final logits
        logits0 = next_logits.expand(B, -1)
        ids0    = move_ids[:, 0]
        mask0   = move_attn[:, 0]

        lp0 = torch.log_softmax(logits0, dim=-1).gather(1, ids0.unsqueeze(1)).squeeze(1)
        token_logp_sum += torch.where(mask0, lp0, torch.zeros_like(lp0))
        token_count    += mask0.to(token_count.dtype)

        # Advance the model with token 0 to get logits for token 1
        cur_ids    = ids0.unsqueeze(1)                      # (B, 1)
        out        = self.model(input_ids=cur_ids, past_key_values=past_b, use_cache=True)
        past_b     = out.past_key_values
        cur_logits = out.logits[:, -1, :]

        # Tokens 1 … max_len−1
        for t in range(1, max_len):
            ids_t  = move_ids[:, t]
            mask_t = move_attn[:, t]

            lp_t = torch.log_softmax(cur_logits, dim=-1).gather(1, ids_t.unsqueeze(1)).squeeze(1)
            token_logp_sum += torch.where(mask_t, lp_t, torch.zeros_like(lp_t))
            token_count    += mask_t.to(token_count.dtype)

            cur_ids    = ids_t.unsqueeze(1)
            out        = self.model(input_ids=cur_ids, past_key_values=past_b, use_cache=True)
            past_b     = out.past_key_values
            cur_logits = out.logits[:, -1, :]

        # ── Step 5: Return mean log-probability per move ───────────────────
        return (token_logp_sum / token_count.clamp_min(1.0)).detach().cpu().tolist()
