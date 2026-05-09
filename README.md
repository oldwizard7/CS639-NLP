# Diagnosing Reasoning Strategy Collapse in Post-trained Language Models

This repository contains the code, experimental outputs, figures, slides, and final report for our CS639 course project at UW-Madison.

## Overview

Large language models can improve reasoning accuracy after supervised fine-tuning or reinforcement-learning-style post-training. However, higher final-answer accuracy does not necessarily mean that the model explores reasoning paths more broadly.

This project studies **reasoning strategy collapse**, where generated reasoning becomes concentrated around fewer repeated patterns or more linear solution paths. We analyze this phenomenon from three complementary perspectives:

1. **Multi-trajectory sampling and diversity analysis**
2. **Sentence-level Thought Anchor analysis**
3. **Graph-based reasoning topology analysis**

Together, these analyses evaluate reasoning models beyond final-answer accuracy by comparing trajectory-level diversity, local causal sensitivity, and global graph structure.

## Methods

### 1. Multi-Trajectory Sampling Analysis

We evaluate model behavior on **MATH-500** by sampling multiple reasoning trajectories per problem. For each model, we compute:

- Pass@1
- Maj@3
- Self-consistency gain
- Pass@3
- Trajectory distinctness
- Average response length

This analysis tests whether post-training improves single-sample accuracy while reducing the diversity benefit of repeated sampling.

### 2. Sentence-Level Thought Anchor Analysis

We apply a black-box Thought Anchor-style analysis on a matched **MMLU slice**. Each reasoning trace is split into chunks, and we remove one chunk at a time before resampling continuations.

We analyze whether each removed chunk acts as:

- a **fragile positive anchor**, whose removal breaks an originally correct trajectory;
- a **repair anchor**, whose removal repairs an originally incorrect trajectory;
- or a **stuck chunk**, where the model remains incorrect after removal.

This reveals whether model errors are locally recoverable or locked into narrow reasoning paths.

### 3. Graph-Based Reasoning Topology

We convert generated MATH reasoning traces into directed semantic graphs. Nodes represent reasoning steps, and edges represent semantic dependencies such as support, contradiction, or independence.

We compare graph-level metrics across model variants:

- number of nodes
- number of edges
- exploration density
- branching ratio
- convergence ratio
- linearity

This analysis studies whether different training stages produce more linear, branched, or interconnected reasoning structures.

## Models

We compare three variants from the Qwen2.5-Math-7B family:

- `Qwen2.5-Math-7B` / Base
- `Qwen2.5-Math-7B-Instruct` / SFT
- `Qwen2.5-Math-7B-Oat-Zero` / DRPO-style post-trained model

We also include GPT-5.4 nano as a closed-source reference model for the multi-trajectory evaluation.

## Datasets

- **MATH-500**: used for multi-trajectory sampling and graph-based topology analysis.
- **MMLU slice**: used for sentence-level Thought Anchor analysis.
