# Transformer Chess Bot

This project implements a chess engine that combines a pretrained transformer language model with classical chess engine techniques. The goal was to explore whether a general-purpose language model can be used as a policy component to guide move selection in a chess engine.

> **Achieved 7th place out of the full class cohort**, earning bonus points for being among the strongest solutions using a small model — and the **best-performing engine built without any fine-tuning**.

## Overview

The engine uses **DistilGPT-2**, a lightweight decoder-only transformer based on the GPT-2 architecture, to evaluate candidate moves given the current board state. The board position is represented using its **FEN (Forsyth–Edwards Notation)** string and inserted into a short prompt. The model then assigns probabilities to possible move tokens, which provides a rough indication of which moves look plausible in that position.

Because a language model alone is not reliable for tactical play, the model is combined with a **beam-based alpha–beta search** that explores promising continuations while keeping the branching factor manageable.

## Key Features

### Transformer-based move evaluation
- Uses **DistilGPT-2** from HuggingFace Transformers — no fine-tuning applied.
- Scores candidate moves using log-probabilities conditioned on a FEN prompt.

### Alpha–beta beam search
- Limits branching to keep the search computationally manageable.
- Uses heuristic move ordering below the root for speed.

### Root policy guidance
- The transformer ranks moves at the root of the search tree.
- Only the most promising moves are explored further by the search.

### Chess heuristics
- Piece values for captures.
- Development principles (knights toward the center, active bishops).
- Simple positional bonuses.

### Opening book
- A small handcrafted opening repertoire to avoid obvious early mistakes.

## Approach

The transformer acts as a **policy model** that proposes and ranks candidate moves, which are then explored more deeply using an **alpha–beta beam search**. Restricting the number of branches at each depth keeps the algorithm computationally efficient.

To better capture tactical sequences, search depth increases automatically when the algorithm encounters **forcing moves** such as checks or captures, since these positions often require deeper calculation.

Alongside the model-guided search, several simple chess heuristics handle cases where the language model's suggestions are less reliable — including basic piece values, encouraging development (knights toward the center, bishops to active diagonals), and discouraging poor pawn moves.

Finally, a small **opening book** is used for the first few moves so the engine can reach reasonable middlegame positions before the transformer-guided search takes over.

## Transformer Architecture

The engine uses **DistilGPT-2**, a causal decoder-only transformer trained via knowledge distillation from GPT-2. With 6 transformer layers (vs. GPT-2's 12), it offers significantly fewer parameters and faster inference — well-suited for scenarios where the model must be evaluated repeatedly, such as ranking candidate moves during search.

Crucially, **no fine-tuning was performed**. The model is used entirely out-of-the-box, relying on patterns learned during general-purpose language model pretraining. The strong results achieved without task-specific training are a key finding of this project.

## Results

- **7th place** in the class tournament
- **Bonus points** awarded for being among the strongest solutions using a small model
- **Best-performing engine with no fine-tuning** across all submitted solutions

## Background

Developed as part of the course *Transformers: Applications in Language and Communications* at Utrecht University.
