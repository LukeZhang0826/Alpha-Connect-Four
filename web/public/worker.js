importScripts("/ort-web.min.js");

// ── constants ──────────────────────────────────────────────────────────────
var ROWS = 6, COLS = 7, C_PUCT = 2, NUM_SEARCHES = 150;

// ── game logic (mirrors connectFour.ts) ───────────────────────────────────
function getValidMoves(board) {
  var v = [];
  for (var c = 0; c < COLS; c++) v.push(board[0][c] === 0);
  return v;
}

function getNextState(board, col) {
  var next = board.map(function(r) { return r.slice(); });
  for (var row = ROWS - 1; row >= 0; row--) {
    if (next[row][col] === 0) { next[row][col] = 1; break; }
  }
  return next;
}

function flipBoard(board) {
  return board.map(function(r) { return r.map(function(v) { return -v; }); });
}

function checkWin(board, col) {
  var row = -1;
  for (var r = 0; r < ROWS; r++) {
    if (board[r][col] !== 0) { row = r; break; }
  }
  if (row === -1) return false;
  var p = board[row][col];
  function cnt(dr, dc) {
    var n = 0;
    for (var i = 1; i < 4; i++) {
      var rr = row + dr * i, cc = col + dc * i;
      if (rr < 0 || rr >= ROWS || cc < 0 || cc >= COLS || board[rr][cc] !== p) break;
      n++;
    }
    return n;
  }
  return cnt(1,0)>=3 || cnt(0,1)+cnt(0,-1)>=3 || cnt(1,1)+cnt(-1,-1)>=3 || cnt(1,-1)+cnt(-1,1)>=3;
}

function isDraw(board) {
  for (var c = 0; c < COLS; c++) if (board[0][c] === 0) return false;
  return true;
}

function getEncodedState(board) {
  var size = ROWS * COLS, enc = new Float32Array(3 * size);
  for (var r = 0; r < ROWS; r++) {
    for (var c = 0; c < COLS; c++) {
      var idx = r * COLS + c, v = board[r][c];
      enc[0 * size + idx] = v === -1 ? 1 : 0;
      enc[1 * size + idx] = v ===  0 ? 1 : 0;
      enc[2 * size + idx] = v ===  1 ? 1 : 0;
    }
  }
  return enc;
}

// ── MCTS ───────────────────────────────────────────────────────────────────
function softmax(arr) {
  var max = -Infinity;
  for (var i = 0; i < arr.length; i++) if (arr[i] > max) max = arr[i];
  var sum = 0, out = new Array(arr.length);
  for (var i = 0; i < arr.length; i++) { out[i] = Math.exp(arr[i] - max); sum += out[i]; }
  for (var i = 0; i < arr.length; i++) out[i] /= sum;
  return out;
}

function makeNode(board, actionTaken, prior, terminal, terminalVal) {
  return { board: board, actionTaken: actionTaken, prior: prior,
           visitCount: 0, valueSum: 0, children: [],
           terminal: terminal, terminalVal: terminalVal };
}

function ucb(parent, child) {
  var q = child.visitCount === 0 ? 0 : 1 - (child.valueSum / child.visitCount + 1) / 2;
  return q + C_PUCT * (Math.sqrt(parent.visitCount) / (child.visitCount + 1)) * child.prior;
}

function selectChild(node) {
  var best = node.children[0], bestScore = -Infinity;
  for (var i = 0; i < node.children.length; i++) {
    var s = ucb(node, node.children[i]);
    if (s > bestScore) { bestScore = s; best = node.children[i]; }
  }
  return best;
}

function expand(node, policy) {
  var valid = getValidMoves(node.board), sum = 0;
  for (var c = 0; c < COLS; c++) if (valid[c]) sum += policy[c];
  for (var c = 0; c < COLS; c++) {
    if (!valid[c]) continue;
    var prior = sum > 0 ? policy[c] / sum : 1 / COLS;
    var childBoard = flipBoard(getNextState(node.board, c));
    var terminal = checkWin(childBoard, c) ? true : isDraw(childBoard);
    var terminalVal = checkWin(childBoard, c) ? 1 : 0;
    // Note: after getNextState+flip, winning means the player who just moved won
    // We detect win on the flipped board by checking the last-placed piece
    var winCheck = false;
    var tmp = getNextState(node.board, c);
    winCheck = checkWin(tmp, c);
    var tv = winCheck ? 1 : 0;
    var term = winCheck || isDraw(tmp);
    node.children.push(makeNode(childBoard, c, prior, term, tv));
  }
}

function backpropagate(path, value) {
  for (var i = path.length - 1; i >= 0; i--) {
    path[i].visitCount++;
    path[i].valueSum += value;
    value = -value;
  }
}

async function runModel(session, board) {
  var enc = getEncodedState(board);
  var tensor = new ort.Tensor("float32", enc, [1, 3, ROWS, COLS]);
  var results = await session.run({ state: tensor });
  var logits = Array.from(results["policy"].data);
  var policy = softmax(logits);
  var value = results["value"].data[0];
  return { policy: policy, value: value };
}

async function mcts(session, board) {
  var root = makeNode(board, null, 0, false, 0);
  root.visitCount = 1;
  var init = await runModel(session, board);
  expand(root, init.policy);

  for (var i = 0; i < NUM_SEARCHES; i++) {
    var path = [root], node = root;
    while (node.children.length > 0 && !node.terminal) {
      node = selectChild(node);
      path.push(node);
    }

    var value;
    if (node.terminal) {
      value = -node.terminalVal;
    } else {
      var out = await runModel(session, node.board);
      expand(node, out.policy);
      value = out.value;
    }
    backpropagate(path, value);
  }

  var bestCol = -1, bestVisits = -1;
  for (var i = 0; i < root.children.length; i++) {
    if (root.children[i].visitCount > bestVisits) {
      bestVisits = root.children[i].visitCount;
      bestCol = root.children[i].actionTaken;
    }
  }
  return bestCol;
}

// ── worker message loop ────────────────────────────────────────────────────
var session = null;

self.onmessage = async function(e) {
  var type = e.data.type, board = e.data.board;

  if (type === "init") {
    try {
      ort.env.wasm.wasmPaths = "/";
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
    if (!session) { self.postMessage({ type: "error", message: "not initialized" }); return; }
    try {
      var col = await mcts(session, board);
      self.postMessage({ type: "move", col: col });
    } catch (err) {
      self.postMessage({ type: "error", message: String(err) });
    }
  }
};
