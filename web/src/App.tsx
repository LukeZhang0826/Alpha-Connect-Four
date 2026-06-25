import { useCallback, useEffect, useRef, useState } from "react";
import type { InferenceSession } from "onnxruntime-web";
import { BoardView } from "./components/Board";
import { Explainer } from "./components/Explainer";
import {
  Board,
  changePerspective,
  checkWin,
  getInitialState,
  getNextState,
  getValidMoves,
  isDraw,
} from "./connectFour";
import { initSession, getAiMove, Difficulty } from "./mcts";
import styles from "./App.module.css";

type Status = "loading" | "error" | "player-turn" | "ai-thinking" | "win" | "draw";
type PlayerColor = 1 | -1;

export default function App() {
  const [board, setBoard] = useState<Board>(getInitialState);
  const [humanPlayer, setHumanPlayer] = useState<PlayerColor>(1);
  const [status, setStatus] = useState<Status>("loading");
  const [lastMove, setLastMove] = useState<number | null>(null);
  const [winner, setWinner] = useState<"you" | "ai" | null>(null);
  const [difficulty, setDifficulty] = useState<Difficulty>("medium");
  const sessionRef = useRef<InferenceSession | null>(null);

  useEffect(() => {
    initSession()
      .then((s) => { sessionRef.current = s; setStatus("player-turn"); })
      .catch((err) => { console.error("ORT init failed:", err); setStatus("error"); });
  }, []);

  const checkTerminal = useCallback(
    (next: Board, col: number, player: PlayerColor): boolean => {
      if (checkWin(next, col)) {
        setWinner(player === humanPlayer ? "you" : "ai");
        setStatus("win");
        return true;
      }
      if (isDraw(next)) { setStatus("draw"); return true; }
      return false;
    },
    [humanPlayer]
  );

  const runAiMove = useCallback(
    async (aiBoard: Board, currentBoard: Board, aiPlayer: PlayerColor) => {
      setStatus("ai-thinking");
      // Yield to let React render the status before any blocking starts
      await new Promise((r) => setTimeout(r, 30));
      try {
        const col = await getAiMove(sessionRef.current!, aiBoard, difficulty);
        const next = getNextState(currentBoard, col, aiPlayer);
        setLastMove(col);
        setBoard(next);
        if (!checkTerminal(next, col, aiPlayer)) setStatus("player-turn");
      } catch (err) {
        console.error("AI move failed:", err);
        setStatus("error");
      }
    },
    [checkTerminal, difficulty]
  );

  const handleColumnClick = useCallback(
    (col: number) => {
      if (status !== "player-turn") return;
      const valid = getValidMoves(board);
      if (!valid[col]) return;

      const next = getNextState(board, col, humanPlayer);
      setLastMove(col);
      setBoard(next);

      if (checkTerminal(next, col, humanPlayer)) return;

      const aiPlayer = -humanPlayer as PlayerColor;
      const aiBoard = humanPlayer === 1 ? changePerspective(next) : next;
      runAiMove(aiBoard, next, aiPlayer);
    },
    [status, board, humanPlayer, checkTerminal, runAiMove]
  );

  const handleReset = useCallback(
    (humanGoesFirst: boolean) => {
      const newBoard = getInitialState();
      const human: PlayerColor = humanGoesFirst ? 1 : -1;
      const aiPlayer = -human as PlayerColor;
      setBoard(newBoard);
      setHumanPlayer(human);
      setLastMove(null);
      setWinner(null);
      if (humanGoesFirst) {
        setStatus("player-turn");
      } else {
        runAiMove(newBoard, newBoard, aiPlayer);
      }
    },
    [runAiMove]
  );

  const validMoves = getValidMoves(board);

  return (
    <div className={styles.app}>
      <h1 className={styles.title}>AlphaZero Connect Four</h1>
      <p className={styles.subtitle}>
        Trained via self-play with MCTS + ResNet (9 residual blocks)
      </p>

      <div className={styles.legend}>
        <span className={styles.humanDot} />
        <span>You ({humanPlayer === 1 ? "Yellow" : "Red"})</span>
        <span className={styles.aiDot} />
        <span>AI ({humanPlayer === 1 ? "Red" : "Yellow"})</span>
      </div>

      <div className={styles.statusBar}>
        {status === "loading" && "Loading model..."}
        {status === "error" && "Failed to load model - check browser console."}
        {status === "player-turn" && "Your turn - click a column."}
        {status === "ai-thinking" && "AI is thinking..."}
        {status === "win" && (winner === "you" ? "You win!" : "AI wins!")}
        {status === "draw" && "Draw!"}
      </div>

      <BoardView
        board={board}
        validMoves={validMoves}
        onColumnClick={handleColumnClick}
        disabled={status !== "player-turn"}
        lastMove={lastMove}
      />

      <div className={styles.difficultyBar}>
        <span className={styles.diffLabel}>Difficulty:</span>
        {(["easy", "medium", "hard"] as Difficulty[]).map((d) => (
          <button
            key={d}
            className={[styles.diffBtn, difficulty === d ? styles.diffBtnActive : ""].join(" ")}
            onClick={() => setDifficulty(d)}
            disabled={status === "ai-thinking"}
          >
            {d.charAt(0).toUpperCase() + d.slice(1)}
          </button>
        ))}
      </div>

      <div className={styles.controls}>
        <button
          className={styles.btn}
          onClick={() => handleReset(true)}
          disabled={status === "loading" || status === "ai-thinking"}
        >
          New game - I go first
        </button>
        <button
          className={styles.btn}
          onClick={() => handleReset(false)}
          disabled={status === "loading" || status === "ai-thinking"}
        >
          New game - AI goes first
        </button>
      </div>

      <Explainer />
    </div>
  );
}
