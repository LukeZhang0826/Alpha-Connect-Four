import * as ort from "onnxruntime-web";
import {
  Board,
  COLS,
  ROWS,
  checkWin,
  getEncodedState,
  getNextState,
  getValidMoves,
  isDraw,
} from "./connectFour";

ort.env.wasm.wasmPaths = "/";

const C_PUCT = 2;
const NUM_SEARCHES = 150;

interface NodeData {
  board: Board;
  actionTaken: number | null;
  prior: number;
  visitCount: number;
  valueSum: number;
  children: NodeData[];
  isTerminal: boolean;
  terminalValue: number;
}

function createNode(
  board: Board,
  actionTaken: number | null,
  prior: number,
  parentAction: number | null
): NodeData {
  let isTerminal = false;
  let terminalValue = 0;
  if (actionTaken !== null) {
    if (checkWin(board, actionTaken)) {
      isTerminal = true;
      terminalValue = 1;
    } else if (isDraw(board)) {
      isTerminal = true;
      terminalValue = 0;
    }
  }
  return {
    board,
    actionTaken: parentAction,
    prior,
    visitCount: 0,
    valueSum: 0,
    children: [],
    isTerminal,
    terminalValue,
  };
}

function ucb(parent: NodeData, child: NodeData): number {
  const q =
    child.visitCount === 0
      ? 0
      : 1 - (child.valueSum / child.visitCount + 1) / 2;
  return (
    q +
    C_PUCT *
      (Math.sqrt(parent.visitCount) / (child.visitCount + 1)) *
      child.prior
  );
}

function select(node: NodeData): NodeData {
  let best = node.children[0];
  let bestScore = -Infinity;
  for (const child of node.children) {
    const score = ucb(node, child);
    if (score > bestScore) {
      bestScore = score;
      best = child;
    }
  }
  return best;
}

function softmax(logits: Float32Array): Float32Array {
  const max = Math.max(...Array.from(logits));
  const exps = logits.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum) as unknown as Float32Array;
}

async function runModel(
  session: ort.InferenceSession,
  board: Board
): Promise<{ policy: Float32Array; value: number }> {
  const encoded = getEncodedState(board);
  const tensor = new ort.Tensor("float32", encoded, [1, 3, ROWS, COLS]);
  const results = await session.run({ state: tensor });

  const policyLogits = results["policy"].data as Float32Array;
  const policy = softmax(policyLogits);
  const value = (results["value"].data as Float32Array)[0];
  return { policy, value };
}

function expand(node: NodeData, policy: Float32Array): void {
  const valid = getValidMoves(node.board);
  let policySum = 0;
  for (let col = 0; col < COLS; col++) {
    if (valid[col]) policySum += policy[col];
  }

  for (let col = 0; col < COLS; col++) {
    if (!valid[col]) continue;
    const prior = policySum > 0 ? policy[col] / policySum : 1 / COLS;
    // After playing col as player 1, flip perspective for opponent
    const childBoard = getNextState(node.board, col, 1);
    const flipped = childBoard.map((row) =>
      row.map((v) => -v)
    ) as Board;
    node.children.push(createNode(flipped, col, prior, col));
  }
}

function backpropagate(path: NodeData[], value: number): void {
  for (let i = path.length - 1; i >= 0; i--) {
    path[i].visitCount++;
    path[i].valueSum += value;
    value = -value;
  }
}

async function mcts(
  session: ort.InferenceSession,
  board: Board
): Promise<number> {
  const root = createNode(board, null, 0, null);
  root.visitCount = 1;

  const { policy: rootPolicy } = await runModel(session, board);
  expand(root, rootPolicy);

  for (let i = 0; i < NUM_SEARCHES; i++) {
    const path: NodeData[] = [root];
    let node = root;

    // Selection
    while (node.children.length > 0 && !node.isTerminal) {
      node = select(node);
      path.push(node);
    }

    let value: number;
    if (node.isTerminal) {
      // value from the perspective of the node that just played
      value = -node.terminalValue;
    } else {
      const { policy, value: netValue } = await runModel(session, node.board);
      expand(node, policy);
      value = netValue;
    }

    backpropagate(path, value);
  }

  // Pick column with highest visit count
  let bestCol = -1;
  let bestVisits = -1;
  for (const child of root.children) {
    if (child.visitCount > bestVisits) {
      bestVisits = child.visitCount;
      bestCol = child.actionTaken!;
    }
  }
  return bestCol;
}

// Worker message interface
let session: ort.InferenceSession | null = null;

self.onmessage = async (e: MessageEvent) => {
  const { type, board } = e.data as { type: string; board: Board };

  if (type === "init") {
    try {
      ort.env.wasm.numThreads = 1;
      session = await ort.InferenceSession.create("/model.onnx", {
        executionProviders: ["wasm"],
      });
      self.postMessage({ type: "ready" });
    } catch (err) {
      self.postMessage({ type: "error", message: String(err) });
    }
    return;
  }

  if (type === "move") {
    if (!session) {
      self.postMessage({ type: "error", message: "Session not initialized" });
      return;
    }
    try {
      const col = await mcts(session, board);
      self.postMessage({ type: "move", col });
    } catch (err) {
      self.postMessage({ type: "error", message: String(err) });
    }
  }
};
