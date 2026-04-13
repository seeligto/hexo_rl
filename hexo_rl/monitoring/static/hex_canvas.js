/**
 * hex_canvas.js — Reusable hex grid renderer for pointy-top axial coordinates.
 *
 * Knows how to render cells, stones, overlays, and coordinate labels.
 * Does NOT know about policies, threats, MCTS, or game state — those are
 * the callers' concerns.
 *
 * Used by /viewer (game replay) and /analyze (policy viewer).
 */

// ── CSS custom property defaults ─────────────────────────────────────────────
const CSS_DEFAULTS = {
  '--player0':       '#378ADD',
  '--player1':       '#D85A30',
  '--last-ring':     '#639922',
  '--grid-stroke':   '#1a3050',
  '--grid-hover':    '#2a4a70',
  '--coord-color':   '#555',
  '--stone-outline':  '#0a0a1a',
  '--canvas-bg':     '#16213e',
  '--empty-text':    '#444',
};

export class HexCanvas {
  /**
   * @param {HTMLElement} containerEl — the element that will hold the canvas
   * @param {Object} [options]
   * @param {string} [options.coordLabels='none'] — 'axial' | 'none'
   * @param {string} [options.orientation='pointy-top']
   * @param {string} [options.emptyText] — text shown when no board is set
   * @param {number} [options.minRadius=0] — auto-bounds mode: ensure grid covers at
   *   least this many axial steps from (0,0) in all directions (useful for empty board)
   */
  constructor(containerEl, options = {}) {
    this._container = containerEl;
    this._canvas = containerEl.tagName === 'CANVAS'
      ? containerEl
      : containerEl.querySelector('canvas') || document.createElement('canvas');
    if (this._canvas.parentElement !== containerEl && containerEl.tagName !== 'CANVAS') {
      containerEl.appendChild(this._canvas);
    }
    this._ctx = this._canvas.getContext('2d');

    // State
    this._stones = [];          // [{q, r, player}]
    this._overlays = new Map(); // name → [{q, r, color?, opacity?, label?, ring?}]
    this._overlayOrder = [];    // insertion order of overlay names
    this._bounds = { mode: 'auto' };
    this._coordLabels = options.coordLabels || 'none';
    this._orientation = options.orientation || 'pointy-top';
    this._emptyText = options.emptyText || null;
    // CHANGE A: minimum radius for auto-bounds empty-board display
    this._minRadius = options.minRadius ?? 0;
    this._clickCb = null;
    this._hoverCoord = null;

    // Computed viewport (set during render)
    this._hexSize = 0;
    this._offX = 0;
    this._offY = 0;
    this._minQ = 0; this._maxQ = 0;
    this._minR = 0; this._maxR = 0;

    // RAF batching
    this._renderQueued = false;

    // Event listeners
    this._onMouseMove = this._handleMouseMove.bind(this);
    this._onMouseLeave = this._handleMouseLeave.bind(this);
    this._onClick = this._handleClick.bind(this);
    this._canvas.addEventListener('mousemove', this._onMouseMove);
    this._canvas.addEventListener('mouseleave', this._onMouseLeave);
    this._canvas.addEventListener('click', this._onClick);
  }

  // ── Public API ─────────────────────────────────────────────────────────────

  /** Replace entire board state. @param {Array<{q,r,player}>} stones */
  setBoard(stones) {
    this._stones = stones || [];
    this._scheduleRender();
  }

  /**
   * Set a named overlay. Replaces existing overlay of same name.
   * @param {string} name
   * @param {Array<{q,r,color?,opacity?,label?,ring?,ringColor?,dash?}>} cells
   */
  setOverlay(name, cells) {
    if (!this._overlays.has(name)) {
      this._overlayOrder.push(name);
    }
    this._overlays.set(name, cells || []);
    this._scheduleRender();
  }

  clearOverlay(name) {
    this._overlays.delete(name);
    this._overlayOrder = this._overlayOrder.filter(n => n !== name);
    this._scheduleRender();
  }

  clearAllOverlays() {
    this._overlays.clear();
    this._overlayOrder = [];
    this._scheduleRender();
  }

  /**
   * @param {{mode: 'radius'|'auto'|'fixed', radius?, minQ?, maxQ?, minR?, maxR?}} bounds
   */
  setBounds(bounds) {
    this._bounds = bounds;
    this._scheduleRender();
  }

  /** @param {'axial'|'none'} mode */
  setCoordLabels(mode) {
    this._coordLabels = mode;
    this._scheduleRender();
  }

  /** @param {'pointy-top'} orientation */
  setOrientation(orientation) {
    this._orientation = orientation;
    this._scheduleRender();
  }

  /** CHANGE C: Update minimum grid radius and re-render. @param {number} r */
  setMinRadius(r) {
    this._minRadius = r;
    this.render();
  }

  /** @param {function({q, r, event})} cb */
  onCellClick(cb) {
    this._clickCb = cb;
  }

  /** Force an immediate synchronous render. */
  render() {
    this._renderQueued = false;
    this._doRender();
  }

  /** Resize canvas to fill container. Call after layout changes. */
  resize() {
    this._scheduleRender();
  }

  /** Read-only access to current viewport for external hit-testing. */
  get hexSize() { return this._hexSize; }
  get offX() { return this._offX; }
  get offY() { return this._offY; }

  destroy() {
    this._canvas.removeEventListener('mousemove', this._onMouseMove);
    this._canvas.removeEventListener('mouseleave', this._onMouseLeave);
    this._canvas.removeEventListener('click', this._onClick);
    this._clickCb = null;
  }

  // ── Hex geometry ───────────────────────────────────────────────────────────

  hexToPixel(q, r) {
    return [
      this._offX + this._hexSize * Math.sqrt(3) * (q + r / 2),
      this._offY + this._hexSize * 1.5 * r,
    ];
  }

  pixelToHex(px, py) {
    const x = px - this._offX;
    const y = py - this._offY;
    const size = this._hexSize;
    if (size === 0) return [0, 0];
    const q = (x * Math.sqrt(3) / 3 - y / 3) / size;
    const r = (y * 2 / 3) / size;
    let rq = Math.round(q), rr = Math.round(r);
    const rs = Math.round(-q - r);
    const dq = Math.abs(rq - q), dr = Math.abs(rr - r), ds = Math.abs(rs - (-q - r));
    if (dq > dr && dq > ds) rq = -rr - rs;
    else if (dr > ds) rr = -rq - rs;
    return [rq, rr];
  }

  // ── Private ────────────────────────────────────────────────────────────────

  _css(prop) {
    const v = getComputedStyle(this._container).getPropertyValue(prop).trim();
    return v || CSS_DEFAULTS[prop] || '';
  }

  _scheduleRender() {
    if (this._renderQueued) return;
    this._renderQueued = true;
    requestAnimationFrame(() => this.render());
  }

  _drawHexPath(ctx, cx, cy, size) {
    ctx.beginPath();
    for (let i = 0; i < 6; i++) {
      const angle = Math.PI / 180 * (60 * i - 30); // pointy-top
      const hx = cx + size * Math.cos(angle);
      const hy = cy + size * Math.sin(angle);
      if (i === 0) ctx.moveTo(hx, hy);
      else ctx.lineTo(hx, hy);
    }
    ctx.closePath();
  }

  _computeBounds() {
    const b = this._bounds;

    // CHANGE D: 'radius' mode is explicit. Auto-mode is the fallback for any
    // unrecognised or absent mode (including when the caller never calls setBounds()).
    if (b.mode === 'radius') {
      let rad = b.radius || 8;

      // CHANGE E: if any stone lies beyond the declared radius, expand to fit.
      // Overflow buffer of +2 prevents stones rendering outside drawn cells.
      for (const s of this._stones) {
        const dist = Math.max(Math.abs(s.q), Math.abs(s.r), Math.abs(s.q + s.r));
        if (dist > rad) rad = dist + 2;
      }

      return { minQ: -rad, maxQ: rad, minR: -rad, maxR: rad };
    }

    if (b.mode === 'fixed') {
      return { minQ: b.minQ || 0, maxQ: b.maxQ || 0, minR: b.minR || 0, maxR: b.maxR || 0 };
    }

    // 'auto' (default) — fit stones + margin, then enforce minRadius.
    // CHANGE B: after computing stone bounds, expand symmetrically so the grid
    // always covers at least _minRadius axial steps from (0,0).
    let minQ, maxQ, minR, maxR;
    if (this._stones.length === 0) {
      // No stones: start from a tiny default and let minRadius expand it below.
      minQ = -1; maxQ = 1; minR = -1; maxR = 1;
    } else {
      minQ = Infinity; maxQ = -Infinity; minR = Infinity; maxR = -Infinity;
      for (const s of this._stones) {
        minQ = Math.min(minQ, s.q); maxQ = Math.max(maxQ, s.q);
        minR = Math.min(minR, s.r); maxR = Math.max(maxR, s.r);
      }
      const margin = b.margin !== undefined ? b.margin : 3;
      minQ -= margin; maxQ += margin;
      minR -= margin; maxR += margin;
    }

    // Enforce minRadius: expand symmetrically from (0,0) if needed.
    if (this._minRadius > 0) {
      minQ = Math.min(minQ, -this._minRadius);
      maxQ = Math.max(maxQ,  this._minRadius);
      minR = Math.min(minR, -this._minRadius);
      maxR = Math.max(maxR,  this._minRadius);
    }

    return { minQ, maxQ, minR, maxR };
  }

  _computeViewport(bounds, canvasW, canvasH) {
    const { minQ, maxQ, minR, maxR } = bounds;
    const testSize = 20;
    let pxMinX = Infinity, pxMaxX = -Infinity, pxMinY = Infinity, pxMaxY = -Infinity;
    for (let q = minQ; q <= maxQ; q++) {
      for (let r = minR; r <= maxR; r++) {
        const px = testSize * Math.sqrt(3) * (q + r / 2);
        const py = testSize * 1.5 * r;
        pxMinX = Math.min(pxMinX, px); pxMaxX = Math.max(pxMaxX, px);
        pxMinY = Math.min(pxMinY, py); pxMaxY = Math.max(pxMaxY, py);
      }
    }
    const pw = pxMaxX - pxMinX + testSize * 4;
    const ph = pxMaxY - pxMinY + testSize * 4;
    const scaleX = canvasW * 0.9 / pw;
    const scaleY = canvasH * 0.9 / ph;
    const scale = Math.min(scaleX, scaleY, 1.5);
    const size = testSize * scale;

    const centerQ = (minQ + maxQ) / 2;
    const centerR = (minR + maxR) / 2;
    const ccx = size * Math.sqrt(3) * (centerQ + centerR / 2);
    const ccy = size * 1.5 * centerR;
    const offX = canvasW / 2 - ccx;
    const offY = canvasH / 2 - ccy;

    return { size, offX, offY };
  }

  _isInBounds(q, r) {
    return q >= this._minQ && q <= this._maxQ && r >= this._minR && r <= this._maxR;
  }

  _doRender() {
    const canvas = this._canvas;
    const ctx = this._ctx;

    // Size canvas to container
    if (canvas.parentElement && canvas.tagName === 'CANVAS') {
      const rect = canvas.parentElement.getBoundingClientRect();
      if (canvas.parentElement.tagName !== 'CANVAS') {
        // If container is not the canvas itself, fit to container
      }
    }
    // Let caller set canvas size externally if needed
    const w = canvas.width;
    const h = canvas.height;

    ctx.clearRect(0, 0, w, h);

    // Empty state
    if (this._stones.length === 0 && this._overlays.size === 0 && this._emptyText) {
      ctx.fillStyle = this._css('--empty-text');
      ctx.font = '16px sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(this._emptyText, w / 2, h / 2);
    }

    // Compute bounds and viewport
    const bounds = this._computeBounds();
    this._minQ = bounds.minQ; this._maxQ = bounds.maxQ;
    this._minR = bounds.minR; this._maxR = bounds.maxR;

    const vp = this._computeViewport(bounds, w, h);
    this._hexSize = vp.size;
    this._offX = vp.offX;
    this._offY = vp.offY;

    const stoneSet = new Set(this._stones.map(s => `${s.q},${s.r}`));
    const p0Color = this._css('--player0');
    const p1Color = this._css('--player1');

    // Layer 1: Empty grid hexes
    ctx.strokeStyle = this._css('--grid-stroke');
    ctx.lineWidth = 0.5;
    for (let q = this._minQ; q <= this._maxQ; q++) {
      for (let r = this._minR; r <= this._maxR; r++) {
        if (stoneSet.has(`${q},${r}`)) continue;
        const [cx, cy] = this.hexToPixel(q, r);
        this._drawHexPath(ctx, cx, cy, vp.size - 1);
        ctx.stroke();

        // Coord labels
        if (this._coordLabels === 'axial' && vp.size > 10) {
          const isHover = this._hoverCoord && this._hoverCoord[0] === q && this._hoverCoord[1] === r;
          ctx.fillStyle = isHover ? '#aaa' : this._css('--coord-color');
          ctx.font = `${Math.max(8, vp.size * 0.35)}px sans-serif`;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(`${q},${r}`, cx, cy);
        }

        // Hover highlight
        if (this._hoverCoord && this._hoverCoord[0] === q && this._hoverCoord[1] === r) {
          this._drawHexPath(ctx, cx, cy, vp.size - 1);
          ctx.fillStyle = 'rgba(255,255,255,0.06)';
          ctx.fill();
        }
      }
    }

    // Layer 2: Overlays in insertion order (skip ring-only overlays, drawn last)
    for (const name of this._overlayOrder) {
      const cells = this._overlays.get(name);
      if (!cells) continue;
      for (const cell of cells) {
        if (cell.ring) continue; // rings drawn in layer 4
        if (stoneSet.has(`${cell.q},${cell.r}`)) continue;
        const [cx, cy] = this.hexToPixel(cell.q, cell.r);

        if (cell.color) {
          const opacity = cell.opacity !== undefined ? cell.opacity : 0.4;
          // Fill hex
          this._drawHexPath(ctx, cx, cy, vp.size - 1);
          ctx.fillStyle = cell.color;
          ctx.globalAlpha = opacity;
          ctx.fill();
          ctx.globalAlpha = 1.0;
        }

        if (cell.circle) {
          // Circle overlay (used for threats)
          ctx.beginPath();
          ctx.arc(cx, cy, vp.size - 3, 0, Math.PI * 2);
          ctx.strokeStyle = cell.circleColor || cell.color || '#fff';
          ctx.lineWidth = cell.circleWidth || 2;
          ctx.setLineDash(cell.dash || []);
          ctx.stroke();
          ctx.setLineDash([]);
          if (cell.circleFill) {
            ctx.fillStyle = cell.circleFill;
            ctx.fill();
          }
        }

        if (cell.label && vp.size > 12) {
          ctx.fillStyle = cell.labelColor || '#fff';
          ctx.font = `${Math.max(8, vp.size * 0.38)}px sans-serif`;
          ctx.textAlign = 'center';
          ctx.textBaseline = 'middle';
          ctx.fillText(cell.label, cx, cy);
        }
      }
    }

    // Layer 3: Stones
    for (const s of this._stones) {
      const [cx, cy] = this.hexToPixel(s.q, s.r);
      this._drawHexPath(ctx, cx, cy, vp.size - 2);
      ctx.fillStyle = s.player === 0 ? p0Color : p1Color;
      ctx.fill();
      ctx.strokeStyle = this._css('--stone-outline');
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    // Layer 4: Ring overlays on top of everything (last_placed, top_k, etc.)
    for (const name of this._overlayOrder) {
      const cells = this._overlays.get(name);
      if (!cells) continue;
      for (const cell of cells) {
        if (!cell.ring) continue;
        const [cx, cy] = this.hexToPixel(cell.q, cell.r);
        ctx.beginPath();
        ctx.arc(cx, cy, vp.size - 4, 0, Math.PI * 2);
        ctx.strokeStyle = cell.ringColor || this._css('--last-ring');
        ctx.lineWidth = cell.ringWidth || 2.5;
        ctx.stroke();
      }
    }
  }

  _handleMouseMove(e) {
    const rect = this._canvas.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;
    if (this._hexSize === 0) return;
    const [q, r] = this.pixelToHex(px, py);
    const prev = this._hoverCoord;
    if (!prev || prev[0] !== q || prev[1] !== r) {
      this._hoverCoord = this._isInBounds(q, r) ? [q, r] : null;
      this._scheduleRender();
    }
  }

  _handleMouseLeave() {
    if (this._hoverCoord) {
      this._hoverCoord = null;
      this._scheduleRender();
    }
  }

  _handleClick(e) {
    if (!this._clickCb || this._hexSize === 0) return;
    const rect = this._canvas.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;
    const [q, r] = this.pixelToHex(px, py);
    if (this._isInBounds(q, r)) {
      this._clickCb({ q, r, event: e });
    }
  }
}
