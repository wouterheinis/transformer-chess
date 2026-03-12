# Transformer Chess Bot

This project implements a chess engine that combines a pretrained transformer language model with classical chess engine techniques. The goal of the project was to explore whether a general-purpose language model can be used as a policy component to guide move selection in a chess engine.

## Overview

The engine uses **DistilGPT-2**, a lightweight **decoder-only transformer** based on the GPT-2 architecture, to evaluate candidate moves given the current board state. The board position is represented using its **FEN (Forsyth–Edwards Notation)** string and inserted into a short prompt. The model then assigns probabilities to possible move tokens, which provides a rough indication of which moves look plausible in that position.

Because a language model alone is not reliable for tactical play, the model is combined with a **beam-based alpha–beta search** that explores promising continuations while keeping the branching factor manageable.

## Key Features

### Transformer-based move evaluation
- Uses **DistilGPT-2** from HuggingFace Transformers.
- Scores candidate moves using log-probabilities conditioned on a FEN prompt.

### Alpha–beta beam search
- Limits branching to keep the search computationally manageable.
- Uses heuristic move ordering below the root for speed.

### Root policy guidance
- The transformer ranks moves at the root of the search tree.
- Only the most promising moves are explored further by the search.

### Basic chess heuristics
- Piece values for captures
- Development principles (knights toward the center, active bishops)
- Simple positional bonuses

### Opening book
- A small handcrafted opening repertoire to avoid obvious early mistakes.

## Approach

The transformer acts as a **policy model** that proposes and ranks candidate moves. These moves are then explored more deeply using an **alpha–beta beam search**. The beam search restricts the number of branches considered at each depth so the algorithm remains computationally efficient.

To better capture tactical sequences, the search depth can increase automatically when the algorithm encounters **forcing moves** such as checks or captures, since these positions often require deeper calculation.

In addition to the model-guided search, I implemented several simple chess heuristics inspired by common beginner principles and my own experience playing chess. These include basic piece values, encouraging development (for example knights toward the center and bishops to active diagonals), and discouraging obviously poor pawn moves. These heuristics help guide the search when the language model’s suggestions are less reliable.

Finally, a small **opening book** is used for the first few moves so the engine can reach reasonable middlegame positions before relying on the transformer-guided search and heuristics.

## Transformer Architecture

The engine uses **DistilGPT-2**, a **causal decoder-only transformer** trained using knowledge distillation from GPT-2. It has **6 transformer layers instead of GPT-2’s 12**, which significantly reduces the number of parameters and makes inference faster. This makes it suitable for scenarios where the model must be evaluated many times, such as ranking candidate moves during search.

## Background

This project was developed as part of an AI assignment for the course Transformers: Applications in Language and Communications at Utrecht University.


