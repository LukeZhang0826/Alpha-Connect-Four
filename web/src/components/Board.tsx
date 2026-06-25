import { Board, COLS, ROWS } from "../connectFour";
import styles from "./Board.module.css";

interface Props {
  board: Board;
  validMoves: boolean[];
  onColumnClick: (col: number) => void;
  disabled: boolean;
  lastMove: number | null;
}

export function BoardView({ board, validMoves, onColumnClick, disabled, lastMove }: Props) {
  return (
    <div className={styles.wrapper}>
      {/* Column hover targets */}
      <div className={styles.colTargets}>
        {Array.from({ length: COLS }, (_, col) => (
          <button
            key={col}
            className={styles.colTarget}
            onClick={() => onColumnClick(col)}
            disabled={disabled || !validMoves[col]}
            aria-label={`Drop piece in column ${col + 1}`}
          />
        ))}
      </div>

      {/* Grid */}
      <div className={styles.grid}>
        {Array.from({ length: ROWS }, (_, row) =>
          Array.from({ length: COLS }, (_, col) => {
            const val = board[row][col];
            const isLast = lastMove === col && row === findTopRow(board, col);
            return (
              <div key={`${row}-${col}`} className={styles.cell}>
                <div
                  className={[
                    styles.piece,
                    val === 1 ? styles.player1 : val === -1 ? styles.player2 : styles.empty,
                    isLast ? styles.lastMove : "",
                  ].join(" ")}
                />
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}

function findTopRow(board: Board, col: number): number {
  for (let r = 0; r < ROWS; r++) {
    if (board[r][col] !== 0) return r;
  }
  return -1;
}
