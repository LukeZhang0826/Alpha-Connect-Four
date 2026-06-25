# Alpha-Connect-Four

A from-scratch [AlphaZero](https://www.science.org/doi/10.1126/science.aar6404) implementation in PyTorch that learns to play **Connect Four** and **Tic-Tac-Toe** purely through self-play (no human game data, no hand-crafted heuristics).

The agent combines **Monte Carlo Tree Search (MCTS)** with a **residual convolutional neural network** that jointly predicts move probabilities (policy head) and expected game outcomes (value head). The network is trained entirely on games it generated itself.

**[Play it in your browser](https://github.com/LukeZhang0826/Alpha-Connect-Four)** (the model runs locally in WASM via ONNX Runtime Web, no server required).

---

## Contents

| Path | Description |
| --- | --- |
| `AlphaZero.ipynb` | Full implementation: game logic, network, MCTS, training loop, evaluation. |
| `model_*_ConnectFour.pt` | Trained Connect Four weights (iterations 0–2). |
| `model_*.pt` | Trained Tic-Tac-Toe weights (iterations 0–2). |
| `optimizer_*_*.pt` | Adam optimizer states for resuming training. |
| `scripts/export_onnx.py` | Exports `model_2_ConnectFour.pt` to `web/public/model.onnx`. |
| `web/` | Vite + React frontend (runs the model in-browser via ONNX Runtime Web). |

## How it works

### Neural network

The `ResNet` takes a 3-plane encoded board (`6×7`, planes: opponent / empty / current player) and produces:

- **Policy head**: softmax distribution over the 7 columns (which move to play)
- **Value head**: scalar in `[-1, 1]` estimating the current player's expected outcome

Architecture: `Conv2D (3→128) + BN + ReLU` start block, followed by 9 residual blocks (each: two `3×3 Conv + BN`, with a skip connection), then the two heads.

### MCTS

At inference, the agent runs 150–600 MCTS simulations before each move:

1. **Selection**: from the root, pick children using the PUCT score: `Q + C * P * sqrt(N_parent) / (N_child + 1)`
2. **Expansion**: at a leaf, run the network's policy head to create child nodes weighted by prior probabilities
3. **Evaluation**: the network's value head estimates the leaf's outcome
4. **Backpropagation**: propagate the value back to the root, flipping sign at each level (opponent's win = our loss)

The final move is chosen proportional to visit counts.

### Training loop

```
self-play (MCTS) → collect (state, policy, outcome) → train ResNet → repeat
```

Each AlphaZero iteration:
1. Play N parallel games against itself using the current network + MCTS
2. Collect `(encoded_state, mcts_visit_distribution, terminal_outcome)` tuples
3. Train the network: cross-entropy loss on policy + MSE loss on value
4. Save checkpoint, repeat

Dirichlet noise is added to the root policy during self-play to ensure exploration.

## Web frontend

The `web/` directory is a static Vite + React + TypeScript app. There is no backend. The ONNX model runs in-browser via [ONNX Runtime Web](https://onnxruntime.ai/docs/get-started/with-javascript/web.html) (WASM backend).

```
web/
  public/
    model.onnx              # exported weights
    ort.wasm.min.js         # ORT runtime (loaded via <script> tag)
    ort-wasm-*.wasm         # WASM binaries
  src/
    connectFour.ts          # game rules (TypeScript port)
    mcts.ts                 # MCTS + ORT inference
    components/Board.tsx    # game board UI
    components/Explainer.tsx # how-it-works section
```

### Run locally

```bash
# 1. Export the ONNX model (requires Python 3.10 + torch)
py -3.10 scripts/export_onnx.py

# 2. Start the dev server
cd web
npm install
npm run dev
```

### Build for deployment

```bash
cd web
npm run build   # outputs to web/dist/
```

Deploy `web/dist/` to any static host (Vercel, GitHub Pages, Netlify).

## Requirements

**Training (Python):**
- Python 3.10
- PyTorch 2.0.1 (CUDA 11.7; falls back to CPU automatically)
- NumPy 1.23.5
- `matplotlib`, `tqdm`
- `kaggle_environments` (evaluation cells only)

```bash
pip install torch numpy matplotlib tqdm kaggle-environments
```

**Frontend (Node):**
- Node.js 20+
- npm 10+

## Acknowledgements

Architecture follows the [AlphaZero](https://www.science.org/doi/10.1126/science.aar6404) and [AlphaGo Zero](https://www.nature.com/articles/nature24270) papers by DeepMind.
