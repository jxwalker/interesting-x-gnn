import { Graph, K } from "./graph";

/** Seed label colours (vivid on dark). */
export const PALETTE: ReadonlyArray<readonly [number, number, number]> = [
  [255, 84, 112], // rose
  [56, 226, 255], // cyan
  [255, 196, 61], // amber
  [176, 132, 255], // violet
];
const GREY: readonly [number, number, number] = [62, 78, 96];

export const cssColour = (k: number): string => {
  const c = PALETTE[k]!;
  return `rgb(${c[0]},${c[1]},${c[2]})`;
};

export function fitCanvas(canvas: HTMLCanvasElement, cssW: number, cssH: number, dpr: number): boolean {
  const bw = Math.max(1, Math.round(cssW * dpr));
  const bh = Math.max(1, Math.round(cssH * dpr));
  if (canvas.width === bw && canvas.height === bh) return false;
  canvas.width = bw;
  canvas.height = bh;
  return true;
}

export interface NodeLook {
  r: Float32Array;
  g: Float32Array;
  b: Float32Array;
  /** 0..1 how coloured the node is. */
  lit: Float32Array;
  /** 0..1 extra blow-out for magnitudes above 1 (sum aggregation). */
  blow: Float32Array;
}

export function makeLook(n: number): NodeLook {
  return {
    r: new Float32Array(n),
    g: new Float32Array(n),
    b: new Float32Array(n),
    lit: new Float32Array(n),
    blow: new Float32Array(n),
  };
}

/** Feature vector → colour: palette mix weighted by features, faded from grey by magnitude. */
export function computeLook(feat: Float32Array, n: number, out: NodeLook): void {
  const BLOW_LOG = Math.log1p(40);
  for (let i = 0; i < n; i++) {
    let m = 0;
    let r = 0;
    let g = 0;
    let b = 0;
    for (let k = 0; k < K; k++) {
      const v = feat[i * K + k]!;
      if (v <= 0) continue;
      const c = PALETTE[k]!;
      m += v;
      r += c[0] * v;
      g += c[1] * v;
      b += c[2] * v;
    }
    const lit = 1 - Math.exp(-2.4 * m);
    const blow = Math.min(1, Math.log1p(Math.max(0, m - 1)) / BLOW_LOG);
    if (m > 1e-6) {
      r /= m;
      g /= m;
      b /= m;
    }
    // Mixed vectors look muddier: boost saturation toward the dominant hue.
    r = GREY[0] + (r - GREY[0]) * lit;
    g = GREY[1] + (g - GREY[1]) * lit;
    b = GREY[2] + (b - GREY[2]) * lit;
    // Blow-out pushes toward white-hot.
    out.r[i] = r + (255 - r) * blow * 0.55;
    out.g[i] = g + (255 - g) * blow * 0.55;
    out.b[i] = b + (255 - b) * blow * 0.55;
    out.lit[i] = lit;
    out.blow[i] = blow;
  }
}

export interface DrawState {
  graph: Graph;
  look: NodeLook;
  /** Source look for pulses (features before the current round). */
  srcLook: NodeLook;
  /** 0..1 pulse travel progress, or -1 when no pulses are in flight. */
  pulse: number;
  /** 0..1 arrival flash after the pulses land, or -1. */
  flash: number;
  /** Per-node change in this round (0..1), drives the arrival ring. */
  delta: Float32Array;
  /** Receptive-field distances (-1 outside); null = no highlight. */
  field: Int32Array | null;
  focus: number;
  hover: number;
  hops: number;
  time: number;
}

const ease = (t: number): number => (t < 0.5 ? 2 * t * t : 1 - (-2 * t + 2) ** 2 / 2);

function rgba(r: number, g: number, b: number, a: number): string {
  return `rgba(${r | 0},${g | 0},${b | 0},${a.toFixed(3)})`;
}

export function nodeRadius(graph: Graph, i: number): number {
  const base = Math.max(3.2, Math.min(6, Math.min(graph.w, graph.ht) / 110));
  return base + Math.sqrt(graph.adj[i]!.length) * 0.9;
}

const GLOW_SCALE = 0.25;
let glowCanvas: HTMLCanvasElement | null = null;
let bgCanvas: HTMLCanvasElement | null = null;

function glowLayer(w: number, h: number): HTMLCanvasElement {
  const gw = Math.max(1, Math.ceil(w * GLOW_SCALE));
  const gh = Math.max(1, Math.ceil(h * GLOW_SCALE));
  if (!glowCanvas) glowCanvas = document.createElement("canvas");
  if (glowCanvas.width !== gw || glowCanvas.height !== gh) {
    glowCanvas.width = gw;
    glowCanvas.height = gh;
  }
  return glowCanvas;
}

function background(w: number, h: number): HTMLCanvasElement {
  const bw = Math.max(1, Math.ceil(w / 2));
  const bh = Math.max(1, Math.ceil(h / 2));
  if (bgCanvas && bgCanvas.width === bw && bgCanvas.height === bh) return bgCanvas;
  bgCanvas = document.createElement("canvas");
  bgCanvas.width = bw;
  bgCanvas.height = bh;
  const c = bgCanvas.getContext("2d");
  if (c) {
    c.fillStyle = "#050a12";
    c.fillRect(0, 0, bw, bh);
    const bloom = c.createRadialGradient(bw / 2, bh / 2, 0, bw / 2, bh / 2, Math.max(bw, bh) * 0.7);
    bloom.addColorStop(0, "rgba(40, 80, 120, 0.22)");
    bloom.addColorStop(1, "rgba(5, 10, 18, 0)");
    c.fillStyle = bloom;
    c.fillRect(0, 0, bw, bh);
  }
  return bgCanvas;
}

export function drawStage(canvas: HTMLCanvasElement, w: number, h: number, dpr: number, s: DrawState): void {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const { graph, look, field } = s;
  const { x, y, n } = graph;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.globalCompositeOperation = "source-over";

  // Background: deep navy with a soft centre bloom (cached per size).
  ctx.drawImage(background(w, h), 0, 0, w, h);

  const inField = (i: number): boolean => !field || field[i]! >= 0;
  const dimOut = field ? 0.28 : 1;

  // Edges
  ctx.lineCap = "round";
  for (const [a, b] of graph.edges) {
    const both = inField(a) && inField(b);
    const lit = Math.min(look.lit[a]!, look.lit[b]!);
    const r = (look.r[a]! + look.r[b]!) / 2;
    const g = (look.g[a]! + look.g[b]!) / 2;
    const bl = (look.b[a]! + look.b[b]!) / 2;
    let alpha = 0.16 + 0.42 * lit;
    let width = 0.8 + lit * 0.9;
    if (field) {
      if (both) {
        alpha = Math.min(0.95, alpha + 0.35);
        width += 0.6;
      } else alpha *= dimOut;
    }
    ctx.strokeStyle = rgba(r, g, bl, alpha);
    ctx.lineWidth = width;
    ctx.beginPath();
    ctx.moveTo(x[a]!, y[a]!);
    ctx.lineTo(x[b]!, y[b]!);
    ctx.stroke();
  }

  // Receptive-field halos (under the glow).
  if (field && s.focus >= 0) {
    for (let i = 0; i < n; i++) {
      const d = field[i]!;
      if (d < 0) continue;
      const rr = nodeRadius(graph, i) + 5 + (s.hops - d) * 0.6;
      ctx.strokeStyle = `rgba(232, 244, 255, ${(0.55 - (d / Math.max(1, s.hops + 1)) * 0.35).toFixed(3)})`;
      ctx.lineWidth = 1;
      ctx.setLineDash([2.5, 3]);
      ctx.lineDashOffset = -s.time * 8;
      ctx.beginPath();
      ctx.arc(x[i]!, y[i]!, rr, 0, Math.PI * 2);
      ctx.stroke();
    }
    ctx.setLineDash([]);
  }

  // Glows: drawn into a quarter-res layer (they're soft anyway), then added on top.
  const gl = glowLayer(w, h);
  const gctx = gl.getContext("2d");
  if (gctx) {
    gctx.setTransform(1, 0, 0, 1, 0, 0);
    gctx.clearRect(0, 0, gl.width, gl.height);
    gctx.globalCompositeOperation = "lighter";
    gctx.setTransform(GLOW_SCALE, 0, 0, GLOW_SCALE, 0, 0);
    for (let i = 0; i < n; i++) {
      const lit = look.lit[i]!;
      if (lit < 0.04) continue;
      const blow = look.blow[i]!;
      const nr = nodeRadius(graph, i);
      const breathe = 1 + 0.08 * Math.sin(s.time * 2.1 + i);
      const gr = nr * (2 + 2.4 * lit + 6 * blow) * breathe;
      const a = (0.2 + 0.35 * blow) * lit * (inField(i) ? 1 : dimOut);
      const grad = gctx.createRadialGradient(x[i]!, y[i]!, nr * 0.4, x[i]!, y[i]!, gr);
      grad.addColorStop(0, rgba(look.r[i]!, look.g[i]!, look.b[i]!, a));
      grad.addColorStop(1, rgba(look.r[i]!, look.g[i]!, look.b[i]!, 0));
      gctx.fillStyle = grad;
      gctx.fillRect(x[i]! - gr, y[i]! - gr, gr * 2, gr * 2);
    }
  }
  ctx.globalCompositeOperation = "lighter";
  ctx.drawImage(gl, 0, 0, w, h);

  // Message pulses travelling along every edge, both directions.
  if (s.pulse >= 0) {
    const src = s.srcLook;
    const t = ease(s.pulse);
    const fade = Math.min(1, s.pulse * 5) * Math.min(1, (1 - s.pulse) * 6 + 0.25);
    for (const [a, b] of graph.edges) {
      for (let dir = 0; dir < 2; dir++) {
        const from = dir === 0 ? a : b;
        const to = dir === 0 ? b : a;
        const lit = src.lit[from]!;
        if (lit < 0.06) continue;
        const px = x[from]! + (x[to]! - x[from]!) * t;
        const py = y[from]! + (y[to]! - y[from]!) * t;
        const alpha = fade * (0.35 + 0.65 * lit) * (field && !(inField(a) && inField(b)) ? 0.35 : 1);
        const pr = 1.4 + 1.6 * lit + 2.2 * src.blow[from]!;
        ctx.fillStyle = rgba(src.r[from]!, src.g[from]!, src.b[from]!, alpha * 0.35);
        ctx.beginPath();
        ctx.arc(px, py, pr * 2.6, 0, Math.PI * 2);
        ctx.fill();
        ctx.fillStyle = rgba(
          src.r[from]! + (255 - src.r[from]!) * 0.5,
          src.g[from]! + (255 - src.g[from]!) * 0.5,
          src.b[from]! + (255 - src.b[from]!) * 0.5,
          alpha,
        );
        ctx.beginPath();
        ctx.arc(px, py, pr, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }

  // Arrival rings when a node's features change.
  if (s.flash >= 0) {
    const f = s.flash;
    for (let i = 0; i < n; i++) {
      const d = s.delta[i]!;
      if (d < 0.08) continue;
      const nr = nodeRadius(graph, i);
      ctx.strokeStyle = rgba(look.r[i]!, look.g[i]!, look.b[i]!, (1 - f) * Math.min(1, d) * 0.85);
      ctx.lineWidth = 1.6;
      ctx.beginPath();
      ctx.arc(x[i]!, y[i]!, nr + 2 + f * 14, 0, Math.PI * 2);
      ctx.stroke();
    }
  }
  ctx.globalCompositeOperation = "source-over";

  // Node cores
  for (let i = 0; i < n; i++) {
    const nr = nodeRadius(graph, i);
    const k = inField(i) ? 1 : 0.4;
    const lit = look.lit[i]!;
    ctx.fillStyle = rgba(look.r[i]!, look.g[i]!, look.b[i]!, k * (0.75 + 0.25 * lit));
    ctx.beginPath();
    ctx.arc(x[i]!, y[i]!, nr, 0, Math.PI * 2);
    ctx.fill();
    // Specular highlight
    if (lit > 0.2) {
      ctx.fillStyle = `rgba(255,255,255,${(0.35 * lit * k).toFixed(3)})`;
      ctx.beginPath();
      ctx.arc(x[i]! - nr * 0.3, y[i]! - nr * 0.3, nr * 0.38, 0, Math.PI * 2);
      ctx.fill();
    } else {
      ctx.strokeStyle = `rgba(150,175,200,${(0.35 * k).toFixed(3)})`;
      ctx.lineWidth = 1;
      ctx.stroke();
    }
  }

  // Seeds: solid white ring + slowly rotating outer arc in the label colour.
  for (let i = 0; i < n; i++) {
    const sd = graph.seed[i]!;
    if (sd < 0) continue;
    const nr = nodeRadius(graph, i);
    const c = PALETTE[sd]!;
    ctx.strokeStyle = "rgba(255,255,255,0.95)";
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.arc(x[i]!, y[i]!, nr + 2.2, 0, Math.PI * 2);
    ctx.stroke();
    const rot = s.time * 1.4 + i;
    ctx.strokeStyle = rgba(c[0], c[1], c[2], 0.9);
    ctx.lineWidth = 2;
    for (let q = 0; q < 3; q++) {
      const a0 = rot + (q * Math.PI * 2) / 3;
      ctx.beginPath();
      ctx.arc(x[i]!, y[i]!, nr + 6, a0, a0 + 1.3);
      ctx.stroke();
    }
  }

  // Focus + hover markers
  if (s.focus >= 0) {
    const nr = nodeRadius(graph, s.focus);
    const pulse = 0.5 + 0.5 * Math.sin(s.time * 4);
    ctx.strokeStyle = `rgba(255,255,255,${(0.5 + 0.4 * pulse).toFixed(3)})`;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.arc(x[s.focus]!, y[s.focus]!, nr + 10 + pulse * 3, 0, Math.PI * 2);
    ctx.stroke();
  }
  if (s.hover >= 0 && s.hover !== s.focus) {
    const nr = nodeRadius(graph, s.hover);
    ctx.strokeStyle = "rgba(255,255,255,0.55)";
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.arc(x[s.hover]!, y[s.hover]!, nr + 5, 0, Math.PI * 2);
    ctx.stroke();
  }
}
