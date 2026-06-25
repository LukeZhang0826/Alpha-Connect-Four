import styles from "./Explainer.module.css";

export function Explainer() {
  return (
    <section className={styles.explainer}>
      <h2 className={styles.heading}>How the AI works</h2>
      <p className={styles.intro}>
        This agent was trained from scratch using <strong>AlphaZero</strong>. No human games,
        no hand-crafted rules. It learned purely by playing against itself, guided by a neural
        network and a search algorithm called MCTS.
      </p>

      <div className={styles.twoCol}>
        {/* ── Neural Network ── */}
        <div className={styles.card}>
          <h3 className={styles.cardTitle}>The Neural Network</h3>
          <p className={styles.cardText}>
            The network takes the current board position and outputs two things simultaneously:
            a <strong>policy</strong> (probability distribution over the 7 columns) and a{" "}
            <strong>value</strong> (expected outcome from -1 to 1). It's a residual convolutional
            network, the same architecture family used in image recognition.
          </p>
          <NetworkDiagram />
        </div>

        {/* ── MCTS ── */}
        <div className={styles.card}>
          <h3 className={styles.cardTitle}>Monte Carlo Tree Search</h3>
          <p className={styles.cardText}>
            Before playing, the AI runs <strong>30–800 simulated futures</strong> depending
            on difficulty. Each simulation walks down a game tree, expands new positions using
            the network's policy, and backpropagates the value estimate back to the root.
            Moves are chosen by how often they were visited.
          </p>
          <MCTSDiagram />
        </div>
      </div>

      {/* ── MCTS phases ── */}
      <h3 className={styles.subheading}>The four MCTS phases</h3>
      <div className={styles.phases}>
        <PhaseCard
          number="1"
          title="Selection"
          color="#3b82f6"
          text="Starting from the root (current board), repeatedly pick the child with the highest UCB score until a leaf is reached."
          formula="UCB = Q + C · P · √N_parent / (N_child + 1)"
        />
        <PhaseCard
          number="2"
          title="Expansion"
          color="#8b5cf6"
          text="At the leaf, run the network's policy head to get move probabilities. Create one child node per legal move, weighted by those priors."
        />
        <PhaseCard
          number="3"
          title="Evaluation"
          color="#ec4899"
          text="The network's value head estimates the position's outcome from the current player's perspective. Returns a scalar between -1 (loss) and +1 (win)."
        />
        <PhaseCard
          number="4"
          title="Backpropagation"
          color="#10b981"
          text="Walk back up to the root, incrementing visit counts and accumulating value estimates along the path. Opponent nodes negate the value."
        />
      </div>

      {/* ── Self-play training ── */}
      <h3 className={styles.subheading}>How it was trained</h3>
      <div className={styles.trainingFlow}>
        <TrainingStep icon="♟" label="Self-play" text="Run MCTS to generate moves. Play full games against itself." />
        <Arrow />
        <TrainingStep icon="📦" label="Collect data" text="Store (board state, MCTS visit counts, game outcome) for each position." />
        <Arrow />
        <TrainingStep icon="🧠" label="Train network" text="Minimise cross-entropy loss on the policy and MSE loss on the value." />
        <Arrow />
        <TrainingStep icon="🔁" label="Repeat" text="The stronger network improves self-play quality, generating better training data." />
      </div>

      {/* ── Architecture specs ── */}
      <h3 className={styles.subheading}>Architecture at a glance</h3>
      <div className={styles.specGrid}>
        <Spec label="Input" value="3 × 6 × 7 tensor" note="Planes: opponent pieces / empty / current player" />
        <Spec label="Backbone" value="9 residual blocks" note="128 hidden channels, 3×3 convolutions" />
        <Spec label="Policy head" value="7 logits → softmax" note="One probability per column" />
        <Spec label="Value head" value="Scalar via tanh" note="Expected outcome in [−1, 1]" />
        <Spec label="Training" value="3 iterations of self-play" note="Each iteration generates new games, then retrains" />
        <Spec label="Search (inference)" value="30 / 200 / 800 simulations" note="Easy / Medium / Hard — selectable in the UI" />
      </div>

      {/* ── App architecture ── */}
      <h3 className={styles.subheading}>How this app is hosted</h3>
      <p className={styles.cardText} style={{ maxWidth: 680, marginBottom: 20 }}>
        There is no server. The model runs entirely in your browser using{" "}
        <strong>ONNX Runtime Web</strong>. The trained PyTorch weights were exported to the
        open ONNX format and are loaded at startup as a static file. The MCTS search loop
        runs in JavaScript on your device. No data leaves your browser.
      </p>
      <div className={styles.hostingFlow}>
        <HostingStep icon="🏗" label="Build" text="Vite + React + TypeScript compiles to a static bundle" />
        <Arrow />
        <HostingStep icon="📦" label="Model" text="PyTorch weights exported to model.onnx via torch.onnx.export" />
        <Arrow />
        <HostingStep icon="🌐" label="Runtime" text="onnxruntime-web runs the ResNet in WASM inside your browser" />
        <Arrow />
        <HostingStep icon="🚀" label="Deploy" text="Served as a static site. No backend, no GPU, free to host." />
      </div>

      {/* ── Repo link ── */}
      <div className={styles.repoBox}>
        <div>
          <div className={styles.repoTitle}>Open source</div>
          <div className={styles.repoNote}>
            Training code, model weights, and this frontend are all on GitHub.
          </div>
        </div>
        <a
          href="https://github.com/LukeZhang0826/Alpha-Connect-Four"
          target="_blank"
          rel="noopener noreferrer"
          className={styles.repoLink}
        >
          View on GitHub
        </a>
      </div>
    </section>
  );
}

// ── Sub-components ────────────────────────────────────────────────────────────

function NetworkDiagram() {
  return (
    <svg viewBox="0 0 280 340" className={styles.diagram} aria-hidden="true">
      {/* Input */}
      <rect x="60" y="10" width="160" height="36" rx="6" fill="#e0e7ff" stroke="#6366f1" strokeWidth="1.5" />
      <text x="140" y="24" textAnchor="middle" fontSize="11" fill="#3730a3" fontWeight="600">Input board state</text>
      <text x="140" y="38" textAnchor="middle" fontSize="9" fill="#6366f1">3 planes × 6 rows × 7 cols</text>

      <line x1="140" y1="46" x2="140" y2="60" stroke="#94a3b8" strokeWidth="1.5" markerEnd="url(#arr)" />

      {/* Start block */}
      <rect x="70" y="60" width="140" height="30" rx="6" fill="#dbeafe" stroke="#3b82f6" strokeWidth="1.5" />
      <text x="140" y="80" textAnchor="middle" fontSize="10" fill="#1e40af">Conv2D 3→128 + BN + ReLU</text>

      <line x1="140" y1="90" x2="140" y2="104" stroke="#94a3b8" strokeWidth="1.5" markerEnd="url(#arr)" />

      {/* Residual blocks */}
      {[0, 1, 2].map((i) => (
        <g key={i}>
          <rect x="70" y={104 + i * 36} width="140" height="28" rx="6" fill="#f3e8ff" stroke="#8b5cf6" strokeWidth="1.5" />
          <text x="140" y={123 + i * 36} textAnchor="middle" fontSize="10" fill="#6d28d9">
            Residual Block {i + 1}
          </text>
          {i < 2 && <line x1="140" y1={132 + i * 36} x2="140" y2={140 + i * 36} stroke="#94a3b8" strokeWidth="1.5" markerEnd="url(#arr)" />}
        </g>
      ))}
      <text x="140" y="219" textAnchor="middle" fontSize="11" fill="#7c3aed">⋮</text>
      <text x="140" y="233" textAnchor="middle" fontSize="9" fill="#94a3b8">9 blocks total</text>

      <line x1="140" y1="238" x2="140" y2="252" stroke="#94a3b8" strokeWidth="1.5" />
      {/* Fork */}
      <line x1="140" y1="252" x2="80" y2="265" stroke="#94a3b8" strokeWidth="1.5" markerEnd="url(#arr)" />
      <line x1="140" y1="252" x2="200" y2="265" stroke="#94a3b8" strokeWidth="1.5" markerEnd="url(#arr)" />

      {/* Policy head */}
      <rect x="18" y="265" width="114" height="30" rx="6" fill="#fce7f3" stroke="#ec4899" strokeWidth="1.5" />
      <text x="75" y="281" textAnchor="middle" fontSize="10" fill="#be185d" fontWeight="600">Policy head</text>
      <text x="75" y="292" textAnchor="middle" fontSize="9" fill="#ec4899">7 move probs</text>

      {/* Value head */}
      <rect x="148" y="265" width="114" height="30" rx="6" fill="#d1fae5" stroke="#10b981" strokeWidth="1.5" />
      <text x="205" y="281" textAnchor="middle" fontSize="10" fill="#065f46" fontWeight="600">Value head</text>
      <text x="205" y="292" textAnchor="middle" fontSize="9" fill="#10b981">outcome in [-1, 1]</text>

      <defs>
        <marker id="arr" markerWidth="6" markerHeight="6" refX="3" refY="3" orient="auto">
          <path d="M0,0 L0,6 L6,3 z" fill="#94a3b8" />
        </marker>
      </defs>
    </svg>
  );
}

function MCTSDiagram() {
  const nodeR = 16;
  // Tree layout: root, 3 children, 2 grandchildren under middle child
  const nodes = [
    { id: "root", x: 140, y: 30, label: "root", color: "#3b82f6", visits: "N=800" },
    { id: "c0", x: 60, y: 110, label: "col 0", color: "#94a3b8", visits: "N=12" },
    { id: "c3", x: 140, y: 110, label: "col 3", color: "#3b82f6", visits: "N=98" },
    { id: "c6", x: 220, y: 110, label: "col 6", color: "#94a3b8", visits: "N=40" },
    { id: "g0", x: 100, y: 200, label: "col 1", color: "#8b5cf6", visits: "N=41" },
    { id: "g1", x: 180, y: 200, label: "col 3", color: "#10b981", visits: "N=57" },
    { id: "leaf", x: 180, y: 290, label: "leaf", color: "#ec4899", visits: "N=1" },
  ];
  const edges = [
    ["root", "c0"], ["root", "c3"], ["root", "c6"],
    ["c3", "g0"], ["c3", "g1"],
    ["g1", "leaf"],
  ];
  const find = (id: string) => nodes.find((n) => n.id === id)!;

  return (
    <svg viewBox="0 0 280 340" className={styles.diagram} aria-hidden="true">
      {/* Edges */}
      {edges.map(([a, b]) => {
        const na = find(a), nb = find(b);
        const sel = ["root-c3", "c3-g1", "g1-leaf"].includes(`${a}-${b}`);
        return (
          <line
            key={`${a}-${b}`}
            x1={na.x} y1={na.y + nodeR} x2={nb.x} y2={nb.y - nodeR}
            stroke={sel ? "#3b82f6" : "#cbd5e1"}
            strokeWidth={sel ? 2.5 : 1.5}
            strokeDasharray={sel ? "none" : "4 3"}
          />
        );
      })}

      {/* Backprop arrow */}
      <path d="M200,274 C240,240 240,160 215,126" fill="none" stroke="#10b981" strokeWidth="2" strokeDasharray="5 3" markerEnd="url(#arrGreen)" />
      <text x="248" y="210" fontSize="9" fill="#10b981" fontWeight="600">back-</text>
      <text x="248" y="221" fontSize="9" fill="#10b981" fontWeight="600">prop</text>

      {/* Nodes */}
      {nodes.map((n) => (
        <g key={n.id}>
          <circle cx={n.x} cy={n.y} r={nodeR} fill={n.color} opacity={n.color === "#94a3b8" ? 0.35 : 1} />
          <text x={n.x} y={n.y + 4} textAnchor="middle" fontSize="8" fill="#fff" fontWeight="600">
            {n.label}
          </text>
          <text x={n.x} y={n.y + nodeR + 11} textAnchor="middle" fontSize="8" fill="#64748b">
            {n.visits}
          </text>
        </g>
      ))}

      {/* Neural net icon at leaf */}
      <rect x="130" y="320" width="100" height="18" rx="4" fill="#fce7f3" stroke="#ec4899" strokeWidth="1" />
      <text x="180" y="333" textAnchor="middle" fontSize="9" fill="#be185d">network evaluation</text>
      <line x1="180" y1="306" x2="180" y2="320" stroke="#ec4899" strokeWidth="1.5" markerEnd="url(#arrPink)" />

      {/* Labels */}
      <text x="8" y="115" fontSize="8" fill="#3b82f6" fontWeight="600">select</text>
      <text x="8" y="205" fontSize="8" fill="#8b5cf6" fontWeight="600">expand</text>
      <text x="8" y="295" fontSize="8" fill="#ec4899" fontWeight="600">eval</text>

      <defs>
        <marker id="arrGreen" markerWidth="6" markerHeight="6" refX="3" refY="3" orient="auto">
          <path d="M0,0 L0,6 L6,3 z" fill="#10b981" />
        </marker>
        <marker id="arrPink" markerWidth="6" markerHeight="6" refX="3" refY="3" orient="auto">
          <path d="M0,0 L0,6 L6,3 z" fill="#ec4899" />
        </marker>
      </defs>
    </svg>
  );
}

function PhaseCard({ number, title, color, text, formula }: {
  number: string; title: string; color: string; text: string; formula?: string;
}) {
  return (
    <div className={styles.phaseCard}>
      <div className={styles.phaseNumber} style={{ background: color }}>{number}</div>
      <div>
        <h4 className={styles.phaseTitle} style={{ color }}>{title}</h4>
        <p className={styles.phaseText}>{text}</p>
        {formula && <code className={styles.formula}>{formula}</code>}
      </div>
    </div>
  );
}

function TrainingStep({ icon, label, text }: { icon: string; label: string; text: string }) {
  return (
    <div className={styles.trainStep}>
      <div className={styles.trainIcon}>{icon}</div>
      <div className={styles.trainLabel}>{label}</div>
      <div className={styles.trainText}>{text}</div>
    </div>
  );
}

function HostingStep({ icon, label, text }: { icon: string; label: string; text: string }) {
  return (
    <div className={styles.hostingStep}>
      <div className={styles.trainIcon}>{icon}</div>
      <div className={styles.trainLabel}>{label}</div>
      <div className={styles.trainText}>{text}</div>
    </div>
  );
}

function Arrow() {
  return <div className={styles.trainArrow}>→</div>;
}

function Spec({ label, value, note }: { label: string; value: string; note: string }) {
  return (
    <div className={styles.spec}>
      <div className={styles.specLabel}>{label}</div>
      <div className={styles.specValue}>{value}</div>
      <div className={styles.specNote}>{note}</div>
    </div>
  );
}
