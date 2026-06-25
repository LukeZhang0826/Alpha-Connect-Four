// ort is loaded as a browser global via <script src="/ort-web.min.js"> in index.html.
// See src/ort-global.d.ts for the type declaration.
import type { InferenceSession } from "onnxruntime-web";
import {
  Board, COLS, ROWS,
  checkWin, isDraw,
  getEncodedState, getNextState, getValidMoves,
} from "./connectFour";

export type Difficulty = "easy" | "medium" | "hard";

interface DifficultyConfig {
  numSearches: number;
  temperature: number; // 0 = greedy; >0 = sample ∝ visits^(1/T)
}

export const DIFFICULTY_CONFIG: Record<Difficulty, DifficultyConfig> = {
  easy:   { numSearches: 30,  temperature: 2.0 },
  medium: { numSearches: 200, temperature: 0   },
  hard:   { numSearches: 800, temperature: 0   },
};

const C_PUCT = 2;

interface MctsNode {
  board: Board;
  actionTaken: number | null;
  prior: number;
  visitCount: number;
  valueSum: number;
  children: MctsNode[];
  terminal: boolean;
  terminalVal: number;
}

function makeNode(board: Board, actionTaken: number | null, prior: number): MctsNode {
  let terminal = false, terminalVal = 0;
  if (actionTaken !== null) {
    if (checkWin(board, actionTaken)) { terminal = true; terminalVal = 1; }
    else if (isDraw(board)) { terminal = true; }
  }
  return { board, actionTaken, prior, visitCount: 0, valueSum: 0, children: [], terminal, terminalVal };
}

function ucb(parent: MctsNode, child: MctsNode): number {
  const q = child.visitCount === 0 ? 0 : 1 - (child.valueSum / child.visitCount + 1) / 2;
  return q + C_PUCT * (Math.sqrt(parent.visitCount) / (child.visitCount + 1)) * child.prior;
}

function selectChild(node: MctsNode): MctsNode {
  let best = node.children[0], bestScore = -Infinity;
  for (const c of node.children) { const s = ucb(node, c); if (s > bestScore) { bestScore = s; best = c; } }
  return best;
}

function softmax(logits: Float32Array): number[] {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) if (logits[i] > max) max = logits[i];
  let sum = 0;
  const out = Array.from(logits).map(v => { const e = Math.exp(v - max); sum += e; return e; });
  return out.map(v => v / sum);
}

function expand(node: MctsNode, policy: number[]): void {
  const valid = getValidMoves(node.board);
  let pSum = 0;
  for (let c = 0; c < COLS; c++) if (valid[c]) pSum += policy[c];
  for (let c = 0; c < COLS; c++) {
    if (!valid[c]) continue;
    const prior = pSum > 0 ? policy[c] / pSum : 1 / COLS;
    const childBoard = getNextState(node.board, c, 1).map(r => r.map(v => -v)) as Board;
    node.children.push(makeNode(childBoard, c, prior));
  }
}

function backpropagate(path: MctsNode[], value: number): void {
  for (let i = path.length - 1; i >= 0; i--) {
    path[i].visitCount++;
    path[i].valueSum += value;
    value = -value;
  }
}

async function runModel(
  session: InferenceSession,
  board: Board
): Promise<{ policy: number[]; value: number }> {
  const enc = getEncodedState(board);
  const tensor = new ort.Tensor("float32", enc, [1, 3, ROWS, COLS]);
  const results = await session.run({ state: tensor });
  const policy = softmax(results["policy"].data as Float32Array);
  const value = (results["value"].data as Float32Array)[0];
  return { policy, value };
}

export async function initSession(): Promise<InferenceSession> {
  ort.env.wasm.wasmPaths = "/";
  ort.env.wasm.numThreads = 1;
  return ort.InferenceSession.create("/model.onnx", {
    executionProviders: ["wasm"],
  });
}

function pickMove(children: MctsNode[], temperature: number): number {
  if (temperature === 0) {
    let best = children[0];
    for (const c of children) if (c.visitCount > best.visitCount) best = c;
    return best.actionTaken!;
  }
  const weights = children.map((c) => Math.pow(c.visitCount, 1 / temperature));
  const total = weights.reduce((a, b) => a + b, 0);
  let r = Math.random() * total;
  for (let i = 0; i < children.length; i++) {
    r -= weights[i];
    if (r <= 0) return children[i].actionTaken!;
  }
  return children[children.length - 1].actionTaken!;
}

export async function getAiMove(
  session: InferenceSession,
  board: Board,
  difficulty: Difficulty = "hard"
): Promise<number> {
  const { numSearches, temperature } = DIFFICULTY_CONFIG[difficulty];
  const root = makeNode(board, null, 0);
  root.visitCount = 1;
  const init = await runModel(session, board);
  expand(root, init.policy);

  for (let i = 0; i < numSearches; i++) {
    const path: MctsNode[] = [root];
    let node = root;
    while (node.children.length > 0 && !node.terminal) {
      node = selectChild(node);
      path.push(node);
    }
    let value: number;
    if (node.terminal) {
      value = -node.terminalVal;
    } else {
      const out = await runModel(session, node.board);
      expand(node, out.policy);
      value = out.value;
    }
    backpropagate(path, value);
  }

  return pickMove(root.children, temperature);
}
