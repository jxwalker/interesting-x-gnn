import "./styles.css";
import { Graph, K, Rng, type Agg } from "./graph";
import { PALETTE, computeLook, cssColour, drawStage, fitCanvas, makeLook, nodeRadius, type NodeLook } from "./viz";

const CREDIT_URL = "https://x.com/slavov_n/status/2109250660851982537";
/** Pulse travel time, then colour settle time (s). */
const PULSE = 0.5;
const SETTLE = 0.26;
const FLASH = 0.55;
/** Auto-play: one round every ~0.9 s. */
const AUTO_PERIOD = 0.9;
const OVERSMOOTH_ROUND = 12;
/** Idle visitors get a fresh graph once the colours have settled. */
const IDLE_RESET_ROUND = 22;
const CLICK_SLOP = 6;

const AGG_NOTES: Record<Agg, string> = {
  mean: "Mean: each node averages itself with its neighbours — colours blend smoothly.",
  sum: "Sum: nothing is normalised, so busy hubs pile up messages and blow out white-hot.",
  max: "Max: each channel keeps only its loudest neighbour — fronts move one hop per round with hard edges.",
};

function mustEl<T extends HTMLElement>(root: ParentNode, selector: string, ctor: new () => T): T {
  const el = root.querySelector(selector);
  if (!(el instanceof ctor)) throw new Error(`Missing or wrong element: ${selector}`);
  return el;
}

function setText(el: HTMLElement, next: string): void {
  if (el.textContent !== next) el.textContent = next;
}

const pct = (v: number): string => `${Math.round(v * 100)}%`;
const ease = (t: number): number => (t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2);

function mount(): void {
  const root = document.getElementById("app");
  if (!(root instanceof HTMLElement)) throw new Error("Missing #app");
  const recordMode = new URLSearchParams(window.location.search).has("record");

  const swatches = PALETTE.map((_, k) => `<i style="background:${cssColour(k)}"></i>`).join("");

  root.innerHTML = `
    <div class="grain" aria-hidden="true"></div>
    <div class="wash" aria-hidden="true"></div>

    <header class="hero">
      <div class="hero-row">
        <p class="brand">Interesting X</p>
        <h1 class="title">Graph neural network playground</h1>
      </div>
      <p class="hint" id="hint">
        <span class="hint-dot" aria-hidden="true"></span>
        <span>Click a node to give it a label, then watch its colour spread as neighbours pass messages. Drag nodes; hover to see what a node can “hear”.</span>
      </p>
    </header>

    <main class="layout">
      <section class="stage" id="stage" aria-label="Graph">
        <canvas id="canvas" role="img" aria-label="A force-directed graph whose node colours spread by message passing"></canvas>
        <div class="chip chip-tl" aria-hidden="true">
          <span class="sw">${swatches}</span><span>labels · click to cycle</span>
        </div>
        <div class="chip chip-tr" aria-live="off">
          <span class="chip-k">layer</span><span class="chip-v" id="stage-round">0</span>
        </div>
        <div class="chip chip-bl" id="field-info" hidden></div>
      </section>

      <aside class="panel" aria-label="Controls and measurements">
        <div class="controls">
          <div class="row">
            <button type="button" class="btn btn-primary" id="btn-pass">Pass messages</button>
            <label class="switch" for="autoplay">
              <input type="checkbox" id="autoplay" checked />
              <span class="track" aria-hidden="true"><span class="knob"></span></span>
              <span>Auto-play</span>
            </label>
          </div>
          <div class="field">
            <label for="agg">Aggregation</label>
            <select id="agg">
              <option value="mean" selected>mean — smooth blend</option>
              <option value="sum">sum — hubs blow out</option>
              <option value="max">max — hard fronts</option>
            </select>
          </div>
          <div class="field">
            <label for="hops">Receptive field <span class="unit" id="hops-val">2 hops</span></label>
            <input id="hops" type="range" min="0" max="6" step="1" value="2" />
            <div class="range-meta"><span>off</span><span>6 hops</span></div>
          </div>
          <button type="button" class="btn" id="btn-reset">New graph</button>
        </div>

        <div class="stats" aria-live="off">
          <div class="stat"><span class="stat-k">Round</span><span class="stat-v v-round" id="stat-round">0</span></div>
          <div class="stat"><span class="stat-k">Coloured</span><span class="stat-v v-col" id="stat-coloured">0%</span></div>
          <div class="stat"><span class="stat-k">Accuracy</span><span class="stat-v v-acc" id="stat-acc">—</span></div>
        </div>

        <div class="card">
          <span class="card-title">One round = one GCN layer</span>
          <p class="formula">h<sub>v</sub>′ = σ(W · AGG({h<sub>u</sub> : u ∈ N(v) ∪ {v}}))</p>
          <p class="note" id="agg-note">${AGG_NOTES.mean}</p>
          <p class="note warn" id="smooth-note" hidden>
            Round ${OVERSMOOTH_ROUND}+: stack this many layers and features stop changing — that's over-smoothing.
            Seeds stay pinned, so borders are a tug-of-war.
          </p>
        </div>

        <p class="credit${recordMode ? " record-hide" : ""}">
          Stochastic block model, ${K} hidden communities; accuracy compares colours with them. Inspired by
          <a href="${CREDIT_URL}" target="_blank" rel="noreferrer">@slavov_n</a>.
        </p>
      </aside>
    </main>
  `;

  const stage = mustEl(root, "#stage", HTMLElement);
  const canvas = mustEl(root, "#canvas", HTMLCanvasElement);
  const btnPass = mustEl(root, "#btn-pass", HTMLButtonElement);
  const autoplay = mustEl(root, "#autoplay", HTMLInputElement);
  const aggSel = mustEl(root, "#agg", HTMLSelectElement);
  const hopsIn = mustEl(root, "#hops", HTMLInputElement);
  const hopsVal = mustEl(root, "#hops-val", HTMLElement);
  const btnReset = mustEl(root, "#btn-reset", HTMLButtonElement);
  const statRound = mustEl(root, "#stat-round", HTMLElement);
  const stageRound = mustEl(root, "#stage-round", HTMLElement);
  const statColoured = mustEl(root, "#stat-coloured", HTMLElement);
  const statAcc = mustEl(root, "#stat-acc", HTMLElement);
  const fieldInfo = mustEl(root, "#field-info", HTMLElement);
  const aggNote = mustEl(root, "#agg-note", HTMLElement);
  const smoothNote = mustEl(root, "#smooth-note", HTMLElement);

  const rng = new Rng((Date.now() ^ (Math.random() * 1e9)) >>> 0);
  const graph = new Graph();
  let look: NodeLook = makeLook(0);
  let srcLook: NodeLook = makeLook(0);
  let display = new Float32Array(0);
  let delta = new Float32Array(0);

  let agg: Agg = "mean";
  let hops = 2;
  let round = 0;
  /** Time into the current round, or -1 when idle. */
  let roundT = -1;
  let flashT = -1;
  let queued = 0;
  let sinceRound = 0;
  let focus = -1;
  let hover = -1;
  let interacted = false;
  let field: Int32Array | null = null;
  let fieldKey = "";

  function stageSize(): { w: number; h: number } {
    return { w: Math.max(200, stage.clientWidth), h: Math.max(200, stage.clientHeight) };
  }

  function updateStats(): void {
    const sc = graph.score();
    setText(statRound, String(round));
    setText(stageRound, String(round));
    setText(statColoured, pct(sc.coloured));
    setText(statAcc, sc.acc === null ? "—" : pct(sc.acc));
    smoothNote.hidden = round < OVERSMOOTH_ROUND;
  }

  function restartFeatures(): void {
    graph.clearFeatures();
    round = 0;
    roundT = -1;
    flashT = -1;
    queued = 0;
    sinceRound = AUTO_PERIOD - 0.35;
    updateStats();
  }

  function newGraph(): void {
    const { w, h } = stageSize();
    const n = w * h < 210_000 ? 66 : 92;
    graph.generate({ n, communities: K, pIn: n < 80 ? 0.24 : 0.19, pOut: n < 80 ? 0.009 : 0.0065 }, w, h, rng);
    look = makeLook(n);
    srcLook = makeLook(n);
    display = new Float32Array(n * K);
    delta = new Float32Array(n);
    // Seed three of the four communities with three different colours; one stays up for grabs.
    const comms = [0, 1, 2, 3].sort(() => rng.next() - 0.5).slice(0, 3);
    const labels = [0, 1, 2, 3].sort(() => rng.next() - 0.5);
    comms.forEach((c, j) => graph.setSeed(graph.seedOf(c, rng), labels[j]!));
    focus = -1;
    hover = -1;
    fieldKey = "";
    restartFeatures();
  }

  function startRound(): void {
    computeLook(graph.h, graph.n, srcLook);
    graph.computeRound(agg);
    for (let i = 0; i < graph.n; i++) {
      let d = 0;
      for (let k = 0; k < K; k++) {
        const a = Math.min(1, graph.h[i * K + k]!);
        const b = Math.min(1, graph.hNext[i * K + k]!);
        d += Math.abs(b - a);
      }
      delta[i] = d;
    }
    roundT = 0;
    sinceRound = 0;
  }

  function requestRound(): void {
    if (roundT < 0) startRound();
    else queued = Math.min(3, queued + 1);
  }

  function noteInteraction(): void {
    interacted = true;
  }

  // ---------------------------------------------------------------------------
  // Controls

  btnPass.addEventListener("click", () => {
    noteInteraction();
    requestRound();
  });
  autoplay.addEventListener("change", () => {
    noteInteraction();
    sinceRound = AUTO_PERIOD;
  });
  aggSel.addEventListener("change", () => {
    noteInteraction();
    agg = aggSel.value as Agg;
    setText(aggNote, AGG_NOTES[agg]);
    // Re-run from the seeds so the new aggregator's character is obvious.
    restartFeatures();
    sinceRound = AUTO_PERIOD - 0.15;
  });
  hopsIn.addEventListener("input", () => {
    noteInteraction();
    hops = Number(hopsIn.value);
    setText(hopsVal, hops === 0 ? "off" : `${hops} hop${hops === 1 ? "" : "s"}`);
    if (focus < 0 && hops > 0) {
      // Pick a seed so the highlight shows up immediately.
      for (let i = 0; i < graph.n; i++) {
        if (graph.seed[i]! >= 0) {
          focus = i;
          break;
        }
      }
    }
  });
  btnReset.addEventListener("click", () => {
    noteInteraction();
    newGraph();
  });

  // ---------------------------------------------------------------------------
  // Canvas: click a node to cycle its label, drag to move it, hover for its receptive field.

  function toStage(ev: PointerEvent): { x: number; y: number } {
    const rect = canvas.getBoundingClientRect();
    return { x: ev.clientX - rect.left, y: ev.clientY - rect.top };
  }

  function pick(p: { x: number; y: number }, touch: boolean): number {
    const i = graph.nearest(p.x, p.y, touch ? 26 : 18);
    return i >= 0 && Math.hypot(graph.x[i]! - p.x, graph.y[i]! - p.y) < nodeRadius(graph, i) + (touch ? 18 : 10) ? i : -1;
  }

  let down: { id: number; node: number; x: number; y: number; moved: boolean } | null = null;
  canvas.addEventListener("pointerdown", (ev) => {
    if (ev.button !== 0) return;
    noteInteraction();
    const p = toStage(ev);
    const node = pick(p, ev.pointerType === "touch");
    down = { id: ev.pointerId, node, x: p.x, y: p.y, moved: false };
    if (node >= 0) {
      graph.pinned = node;
      graph.pinX = graph.x[node]!;
      graph.pinY = graph.y[node]!;
    }
    try {
      canvas.setPointerCapture(ev.pointerId);
    } catch {
      // synthetic events may not be capturable
    }
  });
  canvas.addEventListener("pointermove", (ev) => {
    const p = toStage(ev);
    if (down && down.id === ev.pointerId) {
      if (Math.hypot(p.x - down.x, p.y - down.y) > CLICK_SLOP) down.moved = true;
      if (down.node >= 0 && down.moved) {
        graph.pinX = Math.max(4, Math.min(graph.w - 4, p.x));
        graph.pinY = Math.max(4, Math.min(graph.ht - 4, p.y));
      }
      return;
    }
    if (ev.pointerType === "mouse") hover = pick(p, false);
  });
  const endPointer = (ev: PointerEvent, cancelled: boolean): void => {
    if (!down || down.id !== ev.pointerId) return;
    const d = down;
    down = null;
    graph.pinned = -1;
    if (cancelled || d.moved) return;
    if (d.node >= 0) {
      const label = graph.cycleSeed(d.node);
      if (label < 0) {
        for (let k = 0; k < K; k++) graph.h[d.node * K + k] = 0;
      }
      graph.applySeeds(graph.hNext);
      focus = d.node;
      // Give the new seed a round straight away if nothing is running.
      if (roundT < 0) sinceRound = Math.max(sinceRound, AUTO_PERIOD - 0.25);
      updateStats();
    } else {
      focus = -1;
    }
  };
  canvas.addEventListener("pointerup", (ev) => endPointer(ev, false));
  canvas.addEventListener("pointercancel", (ev) => endPointer(ev, true));
  canvas.addEventListener("pointerleave", () => {
    hover = -1;
  });

  // ---------------------------------------------------------------------------
  // Frame loop

  let last = performance.now();
  let time = 0;
  function frame(now: number): void {
    const dt = Math.min(1 / 30, Math.max(0, (now - last) / 1000));
    last = now;
    time += dt;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const w = canvas.clientWidth;
    const h = canvas.clientHeight;
    fitCanvas(canvas, w, h, dpr);
    graph.resize(Math.max(200, w), Math.max(200, h));
    graph.step(dt);

    // Round scheduling
    sinceRound += dt;
    if (roundT < 0) {
      if (queued > 0) {
        queued--;
        startRound();
      } else if (autoplay.checked && sinceRound >= AUTO_PERIOD) {
        if (!recordMode && !interacted && round >= IDLE_RESET_ROUND) newGraph();
        else startRound();
      }
    }

    let pulse = -1;
    if (roundT >= 0) {
      const prev = roundT;
      roundT += dt;
      if (roundT < PULSE) {
        pulse = roundT / PULSE;
        display.set(graph.h);
      } else {
        if (prev < PULSE) flashT = 0;
        const s = ease(Math.min(1, (roundT - PULSE) / SETTLE));
        for (let j = 0; j < display.length; j++) display[j] = graph.h[j]! + (graph.hNext[j]! - graph.h[j]!) * s;
      }
      if (roundT >= PULSE + SETTLE) {
        graph.commitRound();
        round++;
        roundT = -1;
        updateStats();
      }
    } else {
      display.set(graph.h);
    }
    let flash = -1;
    if (flashT >= 0) {
      flash = flashT / FLASH;
      flashT += dt;
      if (flash >= 1) {
        flashT = -1;
        flash = -1;
      }
    }
    computeLook(display, graph.n, look);

    // Receptive field of the hovered (or last clicked) node.
    const fNode = hover >= 0 ? hover : focus;
    const key = `${fNode}:${hops}`;
    if (key !== fieldKey) {
      fieldKey = key;
      if (fNode >= 0 && hops > 0) {
        field = graph.hopsFrom(fNode, hops);
        let cnt = 0;
        for (let i = 0; i < graph.n; i++) if (field[i]! >= 0) cnt++;
        fieldInfo.hidden = false;
        setText(
          fieldInfo,
          `${hops}-hop receptive field · ${cnt} of ${graph.n} nodes — after ${hops} round${hops === 1 ? "" : "s"} this node has heard only these`,
        );
      } else {
        field = null;
        fieldInfo.hidden = true;
      }
    }

    drawStage(canvas, w, h, dpr, {
      graph,
      look,
      srcLook,
      pulse,
      flash,
      delta,
      field,
      focus: fNode,
      hover,
      hops,
      time,
    });

    requestAnimationFrame(frame);
  }

  newGraph();
  requestAnimationFrame(frame);
}

mount();
