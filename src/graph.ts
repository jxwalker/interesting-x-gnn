/**
 * Stochastic block model graph + force layout + GCN-style message passing.
 * Positions live in CSS pixels of the stage canvas.
 */

export type Agg = "mean" | "sum" | "max";

/** Feature dimensions = number of label colours. */
export const K = 4;

/** Per-aggregator weights: W = self·I − inhibit·(1 − I), plus a hard cap on activations. */
const WEIGHTS: Record<Agg, { gain: number; inhibit: number; cap: number }> = {
  mean: { gain: 1.35, inhibit: 0.45, cap: 1 },
  sum: { gain: 1.0, inhibit: 0.25, cap: 60 },
  max: { gain: 1.0, inhibit: 0.75, cap: 1 },
};

export interface GraphOpts {
  n: number;
  communities: number;
  pIn: number;
  pOut: number;
}

export class Rng {
  private s: number;
  constructor(seed: number) {
    this.s = seed >>> 0 || 1;
  }
  next(): number {
    // mulberry32
    let t = (this.s = (this.s + 0x6d2b79f5) >>> 0);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
  int(n: number): number {
    return Math.floor(this.next() * n);
  }
}

export class Graph {
  n = 0;
  communities = 0;
  /** Hidden ground-truth community of each node. */
  truth: Int32Array = new Int32Array(0);
  edges: Array<[number, number]> = [];
  adj: number[][] = [];
  x = new Float32Array(0);
  y = new Float32Array(0);
  vx = new Float32Array(0);
  vy = new Float32Array(0);
  phase = new Float32Array(0);
  /** Seed label per node: -1 = none, else class index. */
  seed: Int32Array = new Int32Array(0);
  /** Committed features (n × K) and the next round's features. */
  h = new Float32Array(0);
  hNext = new Float32Array(0);
  w = 1;
  ht = 1;
  /** Node index being dragged (pinned to the pointer). */
  pinned = -1;
  pinX = 0;
  pinY = 0;
  private clock = 0;

  generate(opts: GraphOpts, w: number, h: number, rng: Rng): void {
    const { n, communities: c } = opts;
    this.n = n;
    this.communities = c;
    this.w = w;
    this.ht = h;
    this.truth = new Int32Array(n);
    // Slightly uneven community sizes look more natural.
    const sizes = Array.from({ length: c }, () => 0.75 + rng.next() * 0.5);
    const total = sizes.reduce((a, b) => a + b, 0);
    let idx = 0;
    for (let k = 0; k < c; k++) {
      const cnt = k === c - 1 ? n - idx : Math.round((sizes[k]! / total) * n);
      for (let j = 0; j < cnt && idx < n; j++) this.truth[idx++] = k;
    }

    this.edges = [];
    this.adj = Array.from({ length: n }, () => []);
    const link = (a: number, b: number): void => {
      if (a === b || this.adj[a]!.includes(b)) return;
      this.edges.push([a, b]);
      this.adj[a]!.push(b);
      this.adj[b]!.push(a);
    };
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const p = this.truth[i] === this.truth[j] ? opts.pIn : opts.pOut;
        if (rng.next() < p) link(i, j);
      }
    }
    // No isolated or dangling nodes: top up to degree 2 inside the community.
    for (let i = 0; i < n; i++) {
      let guard = 0;
      while (this.adj[i]!.length < 2 && guard++ < 50) {
        const j = rng.int(n);
        if (this.truth[j] === this.truth[i]) link(i, j);
      }
    }
    // Make sure each community touches at least one other (contested borders).
    for (let k = 0; k < c; k++) {
      const a = this.memberOf(k, rng);
      const b = this.memberOf((k + 1) % c, rng);
      link(a, b);
    }

    this.x = new Float32Array(n);
    this.y = new Float32Array(n);
    this.vx = new Float32Array(n);
    this.vy = new Float32Array(n);
    this.phase = new Float32Array(n);
    const cx = w / 2;
    const cy = h / 2;
    const R = Math.min(w, h) * 0.26;
    const rot = rng.next() * Math.PI * 2;
    for (let i = 0; i < n; i++) {
      const k = this.truth[i]!;
      const a = rot + (k / c) * Math.PI * 2;
      const r = Math.min(w, h) * 0.12 * Math.sqrt(rng.next());
      const t = rng.next() * Math.PI * 2;
      this.x[i] = cx + Math.cos(a) * R * (w / Math.max(w, h)) * 1.25 + Math.cos(t) * r;
      this.y[i] = cy + Math.sin(a) * R + Math.sin(t) * r;
      this.phase[i] = rng.next() * 100;
    }
    this.seed = new Int32Array(n).fill(-1);
    this.h = new Float32Array(n * K);
    this.hNext = new Float32Array(n * K);
    this.pinned = -1;
    // Pre-settle so the first frame already looks like a graph.
    for (let s = 0; s < 160; s++) this.step(1 / 60);
  }

  private memberOf(k: number, rng: Rng): number {
    for (let guard = 0; guard < 500; guard++) {
      const i = rng.int(this.n);
      if (this.truth[i] === k) return i;
    }
    return 0;
  }

  /** A typical (median-degree) node of community k, so colour has to travel to the hubs. */
  seedOf(k: number, rng: Rng): number {
    const members: number[] = [];
    for (let i = 0; i < this.n; i++) if (this.truth[i] === k) members.push(i);
    members.sort((a, b) => this.adj[a]!.length - this.adj[b]!.length);
    const mid = Math.floor(members.length * 0.35);
    return members[mid + rng.int(Math.max(1, Math.floor(members.length * 0.3)))] ?? 0;
  }

  resize(w: number, h: number): void {
    if (w === this.w && h === this.ht) return;
    const sx = w / this.w;
    const sy = h / this.ht;
    for (let i = 0; i < this.n; i++) {
      this.x[i]! *= sx;
      this.y[i]! *= sy;
    }
    this.w = w;
    this.ht = h;
  }

  /** Spring length in px for the current stage. */
  restLength(): number {
    return Math.max(16, Math.sqrt((this.w * this.ht) / this.n) * 0.56);
  }

  step(dt: number): void {
    this.clock += dt;
    const { n, x, y, vx, vy } = this;
    const L = this.restLength();
    const fx = new Float32Array(n);
    const fy = new Float32Array(n);
    // Fruchterman–Reingold style: repulsion L²/d, springs d²/L, scaled to px/s².
    const G = 9;
    const rep = G * L * L;
    const cut2 = (L * 7) ** 2;
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        let dx = x[i]! - x[j]!;
        let dy = y[i]! - y[j]!;
        let d2 = dx * dx + dy * dy;
        if (d2 > cut2) continue;
        if (d2 < 1) {
          dx = Math.random() - 0.5;
          dy = Math.random() - 0.5;
          d2 = 1;
        }
        // f/d so (dx, dy)·k is the force vector.
        const k = rep / d2;
        fx[i]! += dx * k;
        fy[i]! += dy * k;
        fx[j]! -= dx * k;
        fy[j]! -= dy * k;
      }
    }
    for (const [a, b] of this.edges) {
      const dx = x[b]! - x[a]!;
      const dy = y[b]! - y[a]!;
      const d = Math.sqrt(dx * dx + dy * dy) || 1;
      const k = (G * d) / L;
      fx[a]! += dx * k;
      fy[a]! += dy * k;
      fx[b]! -= dx * k;
      fy[b]! -= dy * k;
    }
    const cx = this.w / 2;
    const cy = this.ht / 2;
    const pad = Math.min(28, Math.min(this.w, this.ht) * 0.06);
    const t = this.clock;
    // Weak gravity, stronger on the short axis so the graph fills a wide stage.
    const GX = 0.5 * Math.min(1, this.ht / this.w) ** 2;
    const GY = 1.1 * Math.min(1, this.w / this.ht) ** 2;
    for (let i = 0; i < n; i++) {
      // Gentle gravity + a slow personal wander so the graph breathes.
      fx[i]! += (cx - x[i]!) * GX + Math.sin(t * 0.55 + this.phase[i]!) * 14;
      fy[i]! += (cy - y[i]!) * GY + Math.cos(t * 0.47 + this.phase[i]! * 1.3) * 14;
      if (i === this.pinned) {
        x[i] = this.pinX;
        y[i] = this.pinY;
        vx[i] = 0;
        vy[i] = 0;
        continue;
      }
      vx[i] = (vx[i]! + fx[i]! * dt) * 0.9;
      vy[i] = (vy[i]! + fy[i]! * dt) * 0.9;
      const sp = Math.hypot(vx[i]!, vy[i]!);
      const vmax = L * 8;
      if (sp > vmax) {
        vx[i]! *= vmax / sp;
        vy[i]! *= vmax / sp;
      }
      x[i]! += vx[i]! * dt;
      y[i]! += vy[i]! * dt;
      if (x[i]! < pad) (x[i] = pad), (vx[i] = Math.abs(vx[i]!) * 0.5);
      if (x[i]! > this.w - pad) (x[i] = this.w - pad), (vx[i] = -Math.abs(vx[i]!) * 0.5);
      if (y[i]! < pad) (y[i] = pad), (vy[i] = Math.abs(vy[i]!) * 0.5);
      if (y[i]! > this.ht - pad) (y[i] = this.ht - pad), (vy[i] = -Math.abs(vy[i]!) * 0.5);
    }
  }

  nearest(px: number, py: number, maxDist: number): number {
    let best = -1;
    let bd = maxDist * maxDist;
    for (let i = 0; i < this.n; i++) {
      const d = (this.x[i]! - px) ** 2 + (this.y[i]! - py) ** 2;
      if (d < bd) {
        bd = d;
        best = i;
      }
    }
    return best;
  }

  // ---------------------------------------------------------------------------
  // Features / message passing

  clearFeatures(): void {
    this.h.fill(0);
    this.hNext.fill(0);
    this.applySeeds(this.h);
  }

  applySeeds(buf: Float32Array): void {
    for (let i = 0; i < this.n; i++) {
      const s = this.seed[i]!;
      if (s < 0) continue;
      for (let k = 0; k < K; k++) buf[i * K + k] = k === s ? 1 : 0;
    }
  }

  /** Cycle a node's seed label: none → 0 → 1 → … → none. */
  cycleSeed(i: number): number {
    const next = this.seed[i]! + 1;
    this.seed[i] = next >= K ? -1 : next;
    if (this.seed[i]! >= 0) this.applySeeds(this.h);
    return this.seed[i]!;
  }

  setSeed(i: number, label: number): void {
    this.seed[i] = label;
    this.applySeeds(this.h);
  }

  /** One GCN layer: h_v' = σ(W · AGG(h_u, u ∈ N(v) ∪ {v})). Result goes to hNext. */
  computeRound(agg: Agg): void {
    const { n, h, hNext, adj } = this;
    const { gain, inhibit, cap } = WEIGHTS[agg];
    const a = new Float32Array(K);
    for (let v = 0; v < n; v++) {
      const nb = adj[v]!;
      for (let k = 0; k < K; k++) a[k] = h[v * K + k]!;
      if (agg === "max") {
        for (const u of nb) for (let k = 0; k < K; k++) a[k] = Math.max(a[k]!, h[u * K + k]!);
      } else {
        for (const u of nb) for (let k = 0; k < K; k++) a[k]! += h[u * K + k]!;
        if (agg === "mean") for (let k = 0; k < K; k++) a[k]! /= nb.length + 1;
      }
      let tot = 0;
      for (let k = 0; k < K; k++) tot += a[k]!;
      for (let k = 0; k < K; k++) {
        const z = gain * a[k]! - inhibit * (tot - a[k]!);
        // σ = ReLU, clipped so mean/max stay in [0, 1] and sum can blow out.
        hNext[v * K + k] = Math.min(cap, Math.max(0, z));
      }
    }
    this.applySeeds(hNext);
  }

  commitRound(): void {
    this.h.set(this.hNext);
  }

  /** Magnitude of a feature vector in a buffer. */
  static mag(buf: Float32Array, i: number): number {
    let s = 0;
    for (let k = 0; k < K; k++) s += buf[i * K + k]!;
    return s;
  }

  static argmax(buf: Float32Array, i: number): number {
    let best = -1;
    let bv = 0;
    for (let k = 0; k < K; k++) {
      const v = buf[i * K + k]!;
      if (v > bv) {
        bv = v;
        best = k;
      }
    }
    return best;
  }

  /** k-hop BFS distances from a node (-1 = outside). */
  hopsFrom(src: number, hops: number): Int32Array {
    const dist = new Int32Array(this.n).fill(-1);
    if (src < 0) return dist;
    dist[src] = 0;
    let frontier = [src];
    for (let d = 1; d <= hops && frontier.length; d++) {
      const next: number[] = [];
      for (const v of frontier) {
        for (const u of this.adj[v]!) {
          if (dist[u] === -1) {
            dist[u] = d;
            next.push(u);
          }
        }
      }
      frontier = next;
    }
    return dist;
  }

  /**
   * Coloured fraction and accuracy vs the hidden communities. Each colour maps to the
   * community holding most of its seeds; unseeded communities can't be classified right.
   */
  score(): { coloured: number; acc: number | null } {
    const votes = Array.from({ length: K }, () => new Int32Array(this.communities));
    for (let i = 0; i < this.n; i++) {
      const s = this.seed[i]!;
      if (s >= 0) votes[s]![this.truth[i]!]!++;
    }
    const map = votes.map((v) => {
      let best = -1;
      let bv = 0;
      v.forEach((c, k) => {
        if (c > bv) {
          bv = c;
          best = k;
        }
      });
      return best;
    });
    let coloured = 0;
    let right = 0;
    let judged = 0;
    for (let i = 0; i < this.n; i++) {
      if (Graph.mag(this.h, i) < 0.12) continue;
      coloured++;
      if (this.seed[i]! >= 0) continue;
      judged++;
      if (map[Graph.argmax(this.h, i)] === this.truth[i]) right++;
    }
    return { coloured: coloured / Math.max(1, this.n), acc: judged ? right / judged : null };
  }
}
