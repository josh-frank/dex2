/**
 * cvxeda.ts — TypeScript port of cvxEDA (Python original by Luca Citi & Alberto Greco).
 *
 * Original: https://github.com/lciti/cvxEDA
 * Copyright (C) 2014-2025 Luca Citi, Alberto Greco
 *
 * This program is free software; you can redistribute it and/or modify it under
 * the terms of the GNU General Public License as published by the Free Software
 * Foundation; either version 3 of the License, or (at your option) any later
 * version. This program is distributed WITHOUT ANY WARRANTY. See the GNU GPL.
 *
 * This method was first proposed in:
 *   A Greco, G Valenza, A Lanata, EP Scilingo, and L Citi
 *   "cvxEDA: a Convex Optimization Approach to Electrodermal Activity Processing"
 *   IEEE Transactions on Biomedical Engineering, 2015
 *   DOI: 10.1109/TBME.2015.2474131
 *
 * If you use this in support of published research, please cite the reference above.
 *
 * ---------------------------------------------------------------------------
 * Porting notes
 * ---------------------------------------------------------------------------
 * The Python version delegates to cvxopt's QP solver. There is no equivalent in
 * the JS ecosystem, so this file contains a small purpose-built solver:
 *
 *   minimise   0.5 x'Hx + f'x      s.t.   A q >= 0,   x = [q; d; l]
 *
 * It is a Mehrotra predictor-corrector primal-dual interior-point method. The
 * Newton system has an "arrowhead" structure: the q-block is banded
 * (bandwidth 2, from M'M and A'DA) and the d/l-block is small and dense, so it
 * is solved by banded Cholesky + a dense Schur complement. Cost per iteration
 * is roughly O(n·m·bw + m²·w) rather than O(n³).
 *
 * The cvxopt-specific `solver` argument ('conelp' etc.) has no equivalent and
 * is replaced by `solver` tolerances (see CvxEDASolverOptions).
 */

export interface CvxEDASolverOptions {
  /** Relative duality-gap tolerance (default 1e-9, as in the Python version). */
  reltol?: number;
  /** Absolute duality-gap tolerance (default 1e-7). */
  abstol?: number;
  /** Primal/dual feasibility tolerance (default 1e-7). */
  feastol?: number;
  /** Maximum interior-point iterations (default 100). */
  maxIter?: number;
}

export interface CvxEDAOptions {
  /** Slow time constant of the Bateman function (default 2). */
  tau0?: number;
  /** Fast time constant of the Bateman function (default 0.7). */
  tau1?: number;
  /**
   * Knots of the tonic spline. A single number = spacing between knots in
   * seconds; an array = equally spaced knot sample indices. (default 10)
   */
  deltaKnot?: number | ArrayLike<number>;
  /** Penalisation for the sparse SMNA driver (default 8e-4). */
  alpha?: number;
  /** Penalisation for the tonic spline coefficients (default 1e-2). */
  gamma?: number;
  /** 0 = none, 1 = constant, 2 = linear, 'spline_offset' = non-penalised spline offset (default 2). */
  baselineCorrection?: 0 | 1 | 2 | "spline_offset";
  /** Interior-point solver options. */
  solver?: CvxEDASolverOptions;
}

export interface CvxEDAResult {
  /** Phasic component. */
  r: Float64Array;
  /** Sparse SMNA driver of the phasic component. */
  p: Float64Array;
  /** Tonic component. */
  t: Float64Array;
  /** Coefficients of the tonic spline. */
  l: Float64Array;
  /** Offset (and slope) of the baseline drift term. */
  d: Float64Array;
  /** Model residuals. */
  e: Float64Array;
  /** Value of the objective function being minimised (eq. 15 of the paper). */
  obj: number;
  /** Interior-point iterations used. */
  iterations: number;
  /** Whether the solver met its tolerances. */
  converged: boolean;
}

/** A column of a sparse matrix: nonzeros occupy rows r0 .. r0+v.length-1. */
interface Col {
  r0: number;
  v: Float64Array;
}

/** Python-style round-half-to-even. */
function pyRound(x: number): number {
  const f = Math.floor(x);
  const diff = x - f;
  if (diff < 0.5) return f;
  if (diff > 0.5) return f + 1;
  return f % 2 === 0 ? f : f + 1;
}

function dot(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
}

function norm(a: ArrayLike<number>): number {
  return Math.sqrt(dot(a, a));
}

function colDot(a: Col, b: Col): number {
  const lo = Math.max(a.r0, b.r0);
  const hi = Math.min(a.r0 + a.v.length, b.r0 + b.v.length);
  let s = 0;
  for (let i = lo; i < hi; i++) s += a.v[i - a.r0] * b.v[i - b.r0];
  return s;
}

function colDotDense(a: Col, u: ArrayLike<number>): number {
  let s = 0;
  for (let i = 0; i < a.v.length; i++) s += a.v[i] * u[a.r0 + i];
  return s;
}

/** In-place dense Cholesky (lower, row-major). Returns false if not positive definite. */
function denseCholesky(S: Float64Array, m: number): boolean {
  for (let j = 0; j < m; j++) {
    let d = S[j * m + j];
    for (let k = 0; k < j; k++) d -= S[j * m + k] * S[j * m + k];
    if (!(d > 0) || !isFinite(d)) return false;
    const ljj = Math.sqrt(d);
    S[j * m + j] = ljj;
    for (let i = j + 1; i < m; i++) {
      let s = S[i * m + j];
      for (let k = 0; k < j; k++) s -= S[i * m + k] * S[j * m + k];
      S[i * m + j] = s / ljj;
    }
  }
  return true;
}

function denseCholSolve(L: Float64Array, m: number, b: Float64Array, out: Float64Array): void {
  for (let i = 0; i < m; i++) {
    let s = b[i];
    for (let k = 0; k < i; k++) s -= L[i * m + k] * out[k];
    out[i] = s / L[i * m + i];
  }
  for (let i = m - 1; i >= 0; i--) {
    let s = out[i];
    for (let k = i + 1; k < m; k++) s -= L[k * m + i] * out[k];
    out[i] = s / L[i * m + i];
  }
}

export function cvxEDA(yIn: ArrayLike<number>, delta: number, opts: CvxEDAOptions = {}): CvxEDAResult {
  const {
    tau0 = 2.0,
    tau1 = 0.7,
    deltaKnot = 10.0,
    alpha = 8e-4,
    gamma = 1e-2,
    baselineCorrection = 2,
  } = opts;
  const so = { reltol: 1e-9, abstol: 1e-7, feastol: 1e-7, maxIter: 100, ...(opts.solver ?? {}) };

  const n = yIn.length;
  if (n < 3) throw new Error("cvxEDA: signal must have at least 3 samples");
  const y = Float64Array.from(yIn);

  // ---- Bateman ARMA model ----------------------------------------------
  const a1 = 1 / Math.min(tau1, tau0); // a1 > a0
  const a0 = 1 / Math.max(tau1, tau0);
  const den = (a1 - a0) * delta * delta;
  const ar = [
    ((a1 * delta + 2) * (a0 * delta + 2)) / den,
    (2 * a1 * a0 * delta * delta - 8) / den,
    ((a1 * delta - 2) * (a0 * delta - 2)) / den,
  ];
  const ma = [1, 2, 1];

  // Lower-banded Toeplitz operators (bandwidth 2): (Cx)[i] = c0 x[i] + c1 x[i-1] + c2 x[i-2]
  const bandMul = (c: number[], x: Float64Array, out: Float64Array) => {
    for (let i = 0; i < n; i++) {
      let s = c[0] * x[i];
      if (i >= 1) s += c[1] * x[i - 1];
      if (i >= 2) s += c[2] * x[i - 2];
      out[i] = s;
    }
  };
  // Transpose: (C'v)[j] = c0 v[j] + c1 v[j+1] + c2 v[j+2]
  const bandMulT = (c: number[], v: Float64Array, out: Float64Array) => {
    for (let j = 0; j < n; j++) {
      let s = c[0] * v[j];
      if (j + 1 < n) s += c[1] * v[j + 1];
      if (j + 2 < n) s += c[2] * v[j + 2];
      out[j] = s;
    }
  };

  // ---- Spline regressors B ---------------------------------------------
  let knots: number[];
  let deltaKnotS: number;
  if (typeof deltaKnot === "number") {
    deltaKnotS = pyRound(deltaKnot / delta);
    if (deltaKnotS < 1) throw new Error("cvxEDA: deltaKnot must be at least one sample");
    knots = [];
    for (let k = 0; k < n + Math.floor(deltaKnotS / 2); k += deltaKnotS) knots.push(k);
  } else {
    knots = Array.from(deltaKnot);
    if (knots.length < 2) throw new Error("cvxEDA: need at least two knots");
    deltaKnotS = knots[1] - knots[0];
  }
  // order-1 spline convolved with itself, peak-normalised
  const tri: number[] = [];
  for (let v = 1; v < deltaKnotS; v++) tri.push(v);
  for (let v = deltaKnotS; v > 0; v--) tri.push(v);
  const splLen = 2 * tri.length - 1;
  const spl = new Float64Array(splLen);
  for (let i = 0; i < tri.length; i++) for (let j = 0; j < tri.length; j++) spl[i + j] += tri[i] * tri[j];
  const splMax = Math.max(...spl);
  for (let i = 0; i < splLen; i++) spl[i] /= splMax;

  const nB = knots.length;
  const Bcols: Col[] = [];
  for (let k = 0; k < nB; k++) {
    const start = knots[k] - (splLen >> 1);
    const lo = Math.max(0, start);
    const hi = Math.min(n, start + splLen);
    Bcols.push(hi > lo ? { r0: lo, v: spl.slice(lo - start, hi - start) } : { r0: 0, v: new Float64Array(0) });
  }

  // ---- Baseline correction matrix C ------------------------------------
  const Ccols: Col[] = [];
  if (baselineCorrection === "spline_offset") {
    const c = new Float64Array(n);
    for (const col of Bcols) for (let i = 0; i < col.v.length; i++) c[col.r0 + i] += col.v[i];
    Ccols.push({ r0: 0, v: c });
  } else if (baselineCorrection === 0 || baselineCorrection === 1 || baselineCorrection === 2) {
    if (baselineCorrection >= 1) Ccols.push({ r0: 0, v: new Float64Array(n).fill(1) });
    if (baselineCorrection === 2) {
      const c = new Float64Array(n);
      for (let i = 0; i < n; i++) c[i] = (i + 1) / n;
      Ccols.push({ r0: 0, v: c });
    }
  } else {
    throw new Error("cvxEDA: baselineCorrection must be 0, 1, 2 or 'spline_offset'");
  }
  const nC = Ccols.length;

  // W = [C B] : n x m  (columns of the non-q part of the design matrix)
  const W: Col[] = [...Ccols, ...Bcols];
  const m = W.length;

  const applyW = (w: Float64Array, out: Float64Array) => {
    out.fill(0);
    for (let j = 0; j < m; j++) {
      const { r0, v } = W[j];
      const wj = w[j];
      for (let i = 0; i < v.length; i++) out[r0 + i] += v[i] * wj;
    }
  };
  const applyWt = (u: Float64Array, out: Float64Array) => {
    for (let j = 0; j < m; j++) out[j] = colDotDense(W[j], u);
  };

  // ---- Constant parts of the QP ----------------------------------------
  // Hessian H = [M W]'[M W] + gamma * diag(0, I_nB)
  // Linear    f = [alpha*A'1 - M'y ; -W'y]
  const tmpN = new Float64Array(n);
  const ones = new Float64Array(n).fill(1);
  const fq = new Float64Array(n);
  bandMulT(ar, ones, tmpN);
  const MtY = new Float64Array(n);
  bandMulT(ma, y, MtY);
  for (let i = 0; i < n; i++) fq[i] = alpha * tmpN[i] - MtY[i];
  const fw = new Float64Array(m);
  applyWt(y, fw);
  for (let j = 0; j < m; j++) fw[j] = -fw[j];
  const normF = Math.sqrt(dot(fq, fq) + dot(fw, fw));

  // Banded M'M, stored as three diagonals: mm[off][i] = entry (i, i-off)
  const addBand = (band: Float64Array[], c: number[], weight: number | Float64Array) => {
    for (let k = 0; k < n; k++) {
      const wk = typeof weight === "number" ? weight : weight[k];
      for (let a = 0; a < 3; a++) {
        for (let b = a; b < 3; b++) {
          if (k - b < 0) continue;
          band[b - a][k - a] += wk * c[a] * c[b];
        }
      }
    }
  };
  const mm = [new Float64Array(n), new Float64Array(n), new Float64Array(n)];
  addBand(mm, ma, 1);

  // Coupling block Hqw = M'W (n x m), sparse columns
  const Hqw: Col[] = W.map(({ r0, v }) => {
    const lo = Math.max(0, r0 - 2);
    const hi = Math.min(n, r0 + v.length);
    const out = new Float64Array(Math.max(0, hi - lo));
    const get = (k: number) => (k >= r0 && k < r0 + v.length ? v[k - r0] : 0);
    for (let i = lo; i < hi; i++) out[i - lo] = ma[0] * get(i) + ma[1] * get(i + 1) + ma[2] * get(i + 2);
    return { r0: lo, v: out };
  });

  // Hww = W'W + gamma * diag(0, I_nB), dense m x m
  const Hww = new Float64Array(m * m);
  for (let a = 0; a < m; a++) {
    for (let b = a; b < m; b++) {
      const v = colDot(W[a], W[b]);
      Hww[a * m + b] = v;
      Hww[b * m + a] = v;
    }
    if (a >= nC) Hww[a * m + a] += gamma;
  }

  // H x  (given q, w)
  const hxU = new Float64Array(n);
  const hxMq = new Float64Array(n);
  const hxWw = new Float64Array(n);
  const Hx = (q: Float64Array, w: Float64Array, outQ: Float64Array, outW: Float64Array) => {
    bandMul(ma, q, hxMq);
    applyW(w, hxWw);
    for (let i = 0; i < n; i++) hxU[i] = hxMq[i] + hxWw[i];
    bandMulT(ma, hxU, outQ);
    applyWt(hxU, outW);
    for (let j = nC; j < m; j++) outW[j] += gamma * w[j];
  };

  // ---- Interior-point solver -------------------------------------------
  // Variables: q (n), w=[d;l] (m); slack s = A q (n) >= 0; multipliers z (n) >= 0.
  const q = new Float64Array(n);
  const w = new Float64Array(m);
  const s = new Float64Array(n).fill(1);
  const z = new Float64Array(n).fill(1);

  const Hq = new Float64Array(n);
  const Hw = new Float64Array(m);
  const Atz = new Float64Array(n);
  const Aq = new Float64Array(n);
  const rdq = new Float64Array(n);
  const rdw = new Float64Array(m);
  const rp = new Float64Array(n);
  const D = new Float64Array(n);

  // banded Cholesky factor storage for Kqq = M'M + A'DA
  const k = [new Float64Array(n), new Float64Array(n), new Float64Array(n)];
  const L0 = new Float64Array(n);
  const L1 = new Float64Array(n);
  const L2 = new Float64Array(n);
  const Z: Float64Array[] = Array.from({ length: m }, () => new Float64Array(n)); // Kqq^{-1} Hqw
  const S = new Float64Array(m * m);

  const bandFactor = (): boolean => {
    for (let i = 0; i < n; i++) {
      const l2 = i >= 2 ? k[2][i] / L0[i - 2] : 0;
      const l1 = i >= 1 ? (k[1][i] - (i >= 2 ? l2 * L1[i - 1] : 0)) / L0[i - 1] : 0;
      const d = k[0][i] - l1 * l1 - l2 * l2;
      if (!(d > 0) || !isFinite(d)) return false;
      L0[i] = Math.sqrt(d);
      L1[i] = l1;
      L2[i] = l2;
    }
    return true;
  };
  const bandSolve = (b: Float64Array, out: Float64Array) => {
    for (let i = 0; i < n; i++) {
      let v = b[i];
      if (i >= 1) v -= L1[i] * out[i - 1];
      if (i >= 2) v -= L2[i] * out[i - 2];
      out[i] = v / L0[i];
    }
    for (let i = n - 1; i >= 0; i--) {
      let v = out[i];
      if (i + 1 < n) v -= L1[i + 1] * out[i + 1];
      if (i + 2 < n) v -= L2[i + 2] * out[i + 2];
      out[i] = v / L0[i];
    }
  };

  // Solve K [dq; dw] = [rq; rw], K = H + Abar' D Abar
  const solveU = new Float64Array(n);
  const solveRhsW = new Float64Array(m);
  const solveDw = new Float64Array(m);
  const solveK = (rq: Float64Array, rw: Float64Array, dq: Float64Array, dw: Float64Array) => {
    bandSolve(rq, solveU);
    for (let a = 0; a < m; a++) solveRhsW[a] = rw[a] - colDotDense(Hqw[a], solveU);
    denseCholSolve(S, m, solveRhsW, solveDw);
    for (let a = 0; a < m; a++) dw[a] = solveDw[a];
    for (let i = 0; i < n; i++) dq[i] = solveU[i];
    for (let b = 0; b < m; b++) {
      const zb = Z[b];
      const wb = solveDw[b];
      for (let i = 0; i < n; i++) dq[i] -= zb[i] * wb;
    }
  };

  const dq = new Float64Array(n), dw = new Float64Array(m), ds = new Float64Array(n), dz = new Float64Array(n);
  const dqA = new Float64Array(n), dwA = new Float64Array(m), dsA = new Float64Array(n), dzA = new Float64Array(n);
  const rc = new Float64Array(n);
  const vv = new Float64Array(n);
  const rhsQ = new Float64Array(n);
  const rhsW = new Float64Array(m);
  const Adq = new Float64Array(n);

  const direction = (rcv: Float64Array, oq: Float64Array, ow: Float64Array, os: Float64Array, oz: Float64Array) => {
    for (let i = 0; i < n; i++) vv[i] = (rcv[i] - z[i] * rp[i]) / s[i];
    bandMulT(ar, vv, tmpN);
    for (let i = 0; i < n; i++) rhsQ[i] = -rdq[i] + tmpN[i];
    for (let j = 0; j < m; j++) rhsW[j] = -rdw[j];
    solveK(rhsQ, rhsW, oq, ow);
    bandMul(ar, oq, Adq);
    for (let i = 0; i < n; i++) {
      os[i] = Adq[i] + rp[i];
      oz[i] = vv[i] - D[i] * Adq[i];
    }
  };

  const maxStep = (x: Float64Array, dx: Float64Array): number => {
    let a = Infinity;
    for (let i = 0; i < n; i++) if (dx[i] < 0) a = Math.min(a, -x[i] / dx[i]);
    return a;
  };

  let converged = false;
  let iterations = 0;
  for (let iter = 0; iter < so.maxIter; iter++) {
    iterations = iter;
    // residuals
    Hx(q, w, Hq, Hw);
    bandMulT(ar, z, Atz);
    for (let i = 0; i < n; i++) rdq[i] = Hq[i] + fq[i] - Atz[i];
    for (let j = 0; j < m; j++) rdw[j] = Hw[j] + fw[j];
    bandMul(ar, q, Aq);
    for (let i = 0; i < n; i++) rp[i] = Aq[i] - s[i];

    const gap = dot(s, z);
    const mu = gap / n;
    const pobj = 0.5 * (dot(q, Hq) + dot(w, Hw)) + dot(fq, q) + dot(fw, w);
    const pres = norm(rp) / (1 + norm(s));
    const dres = Math.sqrt(dot(rdq, rdq) + dot(rdw, rdw)) / (1 + normF);
    if (pres < so.feastol && dres < so.feastol && (gap < so.abstol || gap < so.reltol * Math.abs(pobj))) {
      converged = true;
      break;
    }

    // Kqq = M'M + A'DA, factorised
    for (let i = 0; i < n; i++) D[i] = z[i] / s[i];
    for (let o = 0; o < 3; o++) k[o].set(mm[o]);
    addBand(k, ar, D);
    if (!bandFactor()) break;

    // Z = Kqq^{-1} Hqw ;  S = Hww - Hqw' Z
    const col = new Float64Array(n);
    for (let b = 0; b < m; b++) {
      col.fill(0);
      col.set(Hqw[b].v, Hqw[b].r0);
      bandSolve(col, Z[b]);
    }
    for (let a = 0; a < m; a++) {
      for (let b = 0; b < m; b++) S[a * m + b] = Hww[a * m + b] - colDotDense(Hqw[a], Z[b]);
    }
    if (!denseCholesky(S, m)) break;

    // predictor (affine-scaling) step
    for (let i = 0; i < n; i++) rc[i] = -s[i] * z[i];
    direction(rc, dqA, dwA, dsA, dzA);
    const aAff = Math.min(1, maxStep(s, dsA), maxStep(z, dzA));
    let muAff = 0;
    for (let i = 0; i < n; i++) muAff += (s[i] + aAff * dsA[i]) * (z[i] + aAff * dzA[i]);
    muAff /= n;
    const sigma = mu > 0 ? Math.pow(Math.max(muAff, 0) / mu, 3) : 0;

    // corrector step
    for (let i = 0; i < n; i++) rc[i] = sigma * mu - s[i] * z[i] - dsA[i] * dzA[i];
    direction(rc, dq, dw, ds, dz);
    const aMax = Math.min(maxStep(s, ds), maxStep(z, dz));
    const step = Math.min(1, 0.99 * aMax);
    if (!(step > 1e-14) || !isFinite(step)) break;

    for (let i = 0; i < n; i++) {
      q[i] += step * dq[i];
      s[i] += step * ds[i];
      z[i] += step * dz[i];
    }
    for (let j = 0; j < m; j++) w[j] += step * dw[j];
  }

  // ---- Assemble outputs --------------------------------------------------
  const d = w.slice(0, nC);
  const l = w.slice(nC);
  const t = new Float64Array(n);
  applyW(w, t);
  const p = new Float64Array(n);
  bandMul(ar, q, p);
  const r = new Float64Array(n);
  bandMul(ma, q, r);
  const e = new Float64Array(n);
  let sse = 0;
  let sumP = 0;
  for (let i = 0; i < n; i++) {
    e[i] = y[i] - r[i] - t[i];
    sse += e[i] * e[i];
    sumP += p[i];
  }
  const obj = 0.5 * sse + 0.5 * gamma * dot(l, l) + alpha * sumP;

  return { r, p, t, l, d, e, obj, iterations, converged };
}
