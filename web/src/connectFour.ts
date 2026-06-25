export const ROWS = 6;
export const COLS = 7;

export type Board = number[][]; // [row][col], values: -1 | 0 | 1

export function getInitialState(): Board {
  return Array.from({ length: ROWS }, () => new Array(COLS).fill(0));
}

export function cloneBoard(board: Board): Board {
  return board.map((row) => [...row]);
}

export function getValidMoves(board: Board): boolean[] {
  return Array.from({ length: COLS }, (_, col) => board[0][col] === 0);
}

export function getNextState(board: Board, col: number, player: 1 | -1): Board {
  const next = cloneBoard(board);
  for (let row = ROWS - 1; row >= 0; row--) {
    if (next[row][col] === 0) {
      next[row][col] = player;
      return next;
    }
  }
  return next;
}

export function checkWin(board: Board, col: number): boolean {
  // Find the piece that was just placed
  let row = -1;
  for (let r = 0; r < ROWS; r++) {
    if (board[r][col] !== 0) {
      row = r;
      break;
    }
  }
  if (row === -1) return false;

  const player = board[row][col];

  function count(dr: number, dc: number): number {
    let n = 0;
    for (let i = 1; i < 4; i++) {
      const r = row + dr * i;
      const c = col + dc * i;
      if (r < 0 || r >= ROWS || c < 0 || c >= COLS || board[r][c] !== player) break;
      n++;
    }
    return n;
  }

  return (
    count(1, 0) >= 3 ||
    count(0, 1) + count(0, -1) >= 3 ||
    count(1, 1) + count(-1, -1) >= 3 ||
    count(1, -1) + count(-1, 1) >= 3
  );
}

export function isDraw(board: Board): boolean {
  return board[0].every((cell) => cell !== 0);
}

/** Encode state as 3-plane float32 array: [opponent, empty, current] (6×7 each) */
export function getEncodedState(board: Board): Float32Array {
  const size = ROWS * COLS;
  const encoded = new Float32Array(3 * size);
  for (let r = 0; r < ROWS; r++) {
    for (let c = 0; c < COLS; c++) {
      const idx = r * COLS + c;
      const v = board[r][c];
      encoded[0 * size + idx] = v === -1 ? 1 : 0; // opponent plane
      encoded[1 * size + idx] = v === 0 ? 1 : 0;  // empty plane
      encoded[2 * size + idx] = v === 1 ? 1 : 0;  // current player plane
    }
  }
  return encoded;
}

/** Flip the board so the AI always reasons as player 1 */
export function changePerspective(board: Board): Board {
  return board.map((row) => row.map((v) => -v as -1 | 0 | 1));
}
