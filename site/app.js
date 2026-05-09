// EMAG interactive viewer.
//   3D: learned Gaussian mixtures (top-K by activation at t).
//   2D: HD topomap of ground truth and EMAG reconstruction at t.
(() => {
  const $ = (id) => document.getElementById(id);

  let state = {
    data: null,
    t: 0,
    playing: false,
    intervalId: null,
    topk: 500,
    clipToHead: true,
    showSphere: true,
    keptIdx: null,
    maxAmpAbs: 1,
    sigmaScale: 1.0,
    ellipsoidOpacity: 0.55,
  };

  const layoutCommon = {
    paper_bgcolor: "#161b22",
    plot_bgcolor: "#161b22",
    font: { color: "#e6edf3" },
    margin: { l: 0, r: 0, t: 0, b: 0 },
  };

  // --- data loading ---------------------------------------------------------

  async function loadIndex() {
    const r = await fetch("data/index.json");
    if (!r.ok) throw new Error("Could not load data/index.json");
    return r.json();
  }

  function recomputeKept() {
    // Filter at the GROUP level: keep a group only if all its Gaussians fit.
    const g = state.data.gaussians;
    if (!state.clipToHead || !state.data.head) {
      state.keptGroups = state.groups.map((_, gi) => gi);
      return;
    }
    const c = state.data.head.center, r = state.data.head.radius;
    const out = [];
    for (let gi = 0; gi < state.groups.length; gi++) {
      let ok = true;
      for (const i of state.groups[gi]) {
        const dx = g.x[i] - c[0], dy = g.y[i] - c[1], dz = g.z[i] - c[2];
        const radial = Math.sqrt(dx*dx + dy*dy + dz*dz);
        const sigma = g.eigvals
          ? Math.sqrt(Math.max(g.eigvals[i][0], g.eigvals[i][1], g.eigvals[i][2]))
          : Math.max(g.sx[i], g.sy[i], g.sz[i]);
        if (radial + state.sigmaScale * sigma > r - 2.0) { ok = false; break; }
      }
      if (ok) out.push(gi);
    }
    state.keptGroups = out;
  }

  async function loadData(file) {
    const r = await fetch(`data/${file}`);
    if (!r.ok) throw new Error(`Could not load data/${file}`);
    state.data = await r.json();
    state.t = 0;
    // Invalidate per-dataset caches: electrode count + signal range differ.
    state.topoGrid = null;
    state.signalScale = null;
    state.maxAmpAbs = Math.max(
      ...state.data.gaussians.amp.map(Math.abs).filter(v => isFinite(v)), 1e-6);

    // Group Gaussians by parent grid point so each mixture renders as one blob.
    const pid = state.data.gaussians.parent_id;
    if (pid) {
      const groups = new Map();
      for (let i = 0; i < pid.length; i++) {
        const p = pid[i];
        if (!groups.has(p)) groups.set(p, []);
        groups.get(p).push(i);
      }
      state.groups = [...groups.values()];   // array of arrays of Gaussian indices
    } else {
      // Fallback: every Gaussian is its own group.
      state.groups = state.data.gaussians.amp.map((_, i) => [i]);
    }
    // Group amplitude bound (sum of |amp| within each group) for static color scale.
    state.maxGroupAmpAbs = Math.max(...state.groups.map(grp =>
      grp.reduce((s, i) => s + Math.abs(state.data.gaussians.amp[i]), 0)
    ), 1e-6);

    recomputeKept();
    $("t").max = state.data.n_time - 1;
    $("t").value = 0;
    $("meta").textContent =
      `${state.groups.length} mixture points (${state.keptGroups.length} in head sphere) · ` +
      `${state.data.n_gaussians} ellipsoids · ${state.data.electrodes.n} electrodes · ` +
      `${state.data.n_time} timesteps · ${state.data.sampling_rate} Hz`;
    redraw(true);
  }

  // --- math helpers ---------------------------------------------------------

  function gaussianActivation(i, tn) {
    const g = state.data.gaussians;
    const dt = (tn - g.mu_t[i]) / Math.max(g.st[i], 1e-4);
    return g.amp[i] * Math.exp(-0.5 * dt * dt);
  }

  // Sum across the G mixture components per group, for keptGroups.
  function groupActivationAtT(tn) {
    const out = new Float32Array(state.keptGroups.length);
    for (let k = 0; k < state.keptGroups.length; k++) {
      const grp = state.groups[state.keptGroups[k]];
      let s = 0;
      for (const i of grp) s += gaussianActivation(i, tn);
      out[k] = s;
    }
    return out;
  }

  function topKLocal(act, k) {
    const N = act.length;
    if (k >= N) return [...Array(N).keys()];
    const arr = new Array(N);
    for (let i = 0; i < N; i++) arr[i] = [Math.abs(act[i]), i];
    arr.sort((a, b) => b[0] - a[0]);
    return arr.slice(0, k).map(([, i]) => i);
  }

  function signalAt(tIdx, which) {
    const arr = state.data[which];
    const scale = state.data.signal_scale;
    const N = arr.length;
    const out = new Float32Array(N);
    for (let e = 0; e < N; e++) out[e] = arr[e][tIdx] * scale;
    return out;
  }

  // Pre-compute a unit sphere mesh once (reused for every ellipsoid).
  function unitSphere(nLat = 12, nLon = 16) {
    const x = [], y = [], z = [], i = [], j = [], k = [];
    for (let lat = 0; lat <= nLat; lat++) {
      const theta = Math.PI * lat / nLat;
      for (let lon = 0; lon <= nLon; lon++) {
        const phi = 2 * Math.PI * lon / nLon;
        x.push(Math.sin(theta) * Math.cos(phi));
        y.push(Math.sin(theta) * Math.sin(phi));
        z.push(Math.cos(theta));
      }
    }
    const W = nLon + 1;
    for (let lat = 0; lat < nLat; lat++) {
      for (let lon = 0; lon < nLon; lon++) {
        const a = lat * W + lon, b = a + 1, c = a + W, d = c + 1;
        i.push(a, a); j.push(b, d); k.push(d, c);
      }
    }
    return { x, y, z, i, j, k, n: x.length };
  }
  const UNIT = unitSphere();

  // Build a single big mesh3d combining many ellipsoids (vertex/face concatenation).
  // For each Gaussian: vert = center + (V * diag(sqrt(eigvals)) * sigmaScale) * unit_sphere_vert
  function ellipsoidsMesh(idxList, sigmaScale, colorVals) {
    const g = state.data.gaussians;
    const evec = g.eigvecs, eval_ = g.eigvals;
    const nUnit = UNIT.n;
    const totalV = idxList.length * nUnit;
    const xs = new Float32Array(totalV);
    const ys = new Float32Array(totalV);
    const zs = new Float32Array(totalV);
    const vc = new Float32Array(totalV);

    let i = 0;
    for (let gi = 0; gi < idxList.length; gi++) {
      const idx = idxList[gi];
      const cx = g.x[idx], cy = g.y[idx], cz = g.z[idx];
      const V = evec[idx];          // 3x3
      const e = eval_[idx];         // 3
      const s0 = Math.sqrt(e[0]) * sigmaScale;
      const s1 = Math.sqrt(e[1]) * sigmaScale;
      const s2 = Math.sqrt(e[2]) * sigmaScale;
      // A = V * diag(sqrt(eigvals)) — column k of A = V[:, k] * s_k
      const A00 = V[0][0]*s0, A01 = V[0][1]*s1, A02 = V[0][2]*s2;
      const A10 = V[1][0]*s0, A11 = V[1][1]*s1, A12 = V[1][2]*s2;
      const A20 = V[2][0]*s0, A21 = V[2][1]*s1, A22 = V[2][2]*s2;
      const c   = colorVals[gi];
      for (let v = 0; v < nUnit; v++) {
        const ux = UNIT.x[v], uy = UNIT.y[v], uz = UNIT.z[v];
        xs[i] = cx + A00*ux + A01*uy + A02*uz;
        ys[i] = cy + A10*ux + A11*uy + A12*uz;
        zs[i] = cz + A20*ux + A21*uy + A22*uz;
        vc[i] = c;
        i++;
      }
    }

    // Faces: replicate UNIT.{i,j,k} with offset per Gaussian
    const fLen = UNIT.i.length * idxList.length;
    const fi = new Int32Array(fLen), fj = new Int32Array(fLen), fk = new Int32Array(fLen);
    let f = 0;
    for (let gi = 0; gi < idxList.length; gi++) {
      const off = gi * nUnit;
      for (let m = 0; m < UNIT.i.length; m++) {
        fi[f] = UNIT.i[m] + off; fj[f] = UNIT.j[m] + off; fk[f] = UNIT.k[m] + off;
        f++;
      }
    }
    return { x: xs, y: ys, z: zs, i: fi, j: fj, k: fk, intensity: vc };
  }

  // Generate icosphere-ish sphere mesh for the head outline
  function sphereMesh(center, radius, nLat = 18, nLon = 28) {
    const x = [], y = [], z = [], i = [], j = [], k = [];
    for (let lat = 0; lat <= nLat; lat++) {
      const theta = Math.PI * lat / nLat;
      for (let lon = 0; lon <= nLon; lon++) {
        const phi = 2 * Math.PI * lon / nLon;
        x.push(center[0] + radius * Math.sin(theta) * Math.cos(phi));
        y.push(center[1] + radius * Math.sin(theta) * Math.sin(phi));
        z.push(center[2] + radius * Math.cos(theta));
      }
    }
    const W = nLon + 1;
    for (let lat = 0; lat < nLat; lat++) {
      for (let lon = 0; lon < nLon; lon++) {
        const a = lat * W + lon, b = a + 1, c = a + W, d = c + 1;
        i.push(a, a); j.push(b, c); k.push(c, d); // two triangles per quad: (a,b,c) and (a,c,d)... wait swap
      }
    }
    // (Re)build i/j/k properly: (a,b,d) + (a,d,c)
    i.length = 0; j.length = 0; k.length = 0;
    for (let lat = 0; lat < nLat; lat++) {
      for (let lon = 0; lon < nLon; lon++) {
        const a = lat * W + lon, b = a + 1, c = a + W, d = c + 1;
        i.push(a, a); j.push(b, d); k.push(d, c);
      }
    }
    return { x, y, z, i, j, k };
  }

  // --- plot builders --------------------------------------------------------

  function build3DTraces() {
    const g = state.data.gaussians;
    const tnDen = Math.max(state.data.time_axis[state.data.n_time - 1], 1);
    const tn = state.data.time_axis[state.t] / tnDen;

    // Pick top-K *groups* by group activation magnitude, then expand to per-Gaussian indices,
    // assigning every Gaussian in a group the SAME group color (mixture renders as one blob).
    const grpAct = groupActivationAtT(tn);
    const topK = topKLocal(grpAct, state.topk);

    const globalIdx = [];      // expanded per-Gaussian indices
    const colors = [];         // per-Gaussian color (= group color, repeated)
    for (const ki of topK) {
      const grp = state.groups[state.keptGroups[ki]];
      const c = grpAct[ki] / state.maxGroupAmpAbs;
      for (const i of grp) {
        globalIdx.push(i);
        colors.push(c);
      }
    }

    let traces = [];
    if (g.eigvals && g.eigvecs && globalIdx.length > 0) {
      const mesh = ellipsoidsMesh(globalIdx, state.sigmaScale, colors);
      traces.push({
        type: "mesh3d",
        x: mesh.x, y: mesh.y, z: mesh.z,
        i: mesh.i, j: mesh.j, k: mesh.k,
        intensity: mesh.intensity,
        intensitymode: "vertex",
        colorscale: "RdBu", reversescale: true,
        cmin: -1, cmax: 1,
        opacity: state.ellipsoidOpacity,
        flatshading: false,
        lighting: { ambient: 0.6, diffuse: 0.6, specular: 0.1, fresnel: 0.1 },
        hoverinfo: "skip", showlegend: false,
      });
    } else if (globalIdx.length > 0) {
      const xs = globalIdx.map(i => g.x[i]);
      const ys = globalIdx.map(i => g.y[i]);
      const zs = globalIdx.map(i => g.z[i]);
      const sizes = globalIdx.map(i => Math.min(28, Math.max(3, g.sx[i] * 0.6)));
      traces.push({
        type: "scatter3d", mode: "markers",
        x: xs, y: ys, z: zs,
        marker: { size: sizes, color: colors, colorscale: "RdBu",
                  reversescale: true, cmin: -1, cmax: 1, opacity: 0.9 },
        hoverinfo: "skip", showlegend: false,
      });
    }

    if (state.showSphere && state.data.head) {
      const m = sphereMesh(state.data.head.center, state.data.head.radius);
      traces.push({
        type: "mesh3d",
        x: m.x, y: m.y, z: m.z, i: m.i, j: m.j, k: m.k,
        opacity: 0.08, color: "#9da7b3",
        flatshading: false, hoverinfo: "skip", showlegend: false,
        lighting: { ambient: 0.85, diffuse: 0.5, specular: 0.0 },
      });
    }
    return { traces, groupCount: topK.length, gaussCount: globalIdx.length };
  }

  // --- smooth interpolated topomap ---------------------------------------
  // Inverse-distance-weighted interpolation onto a regular grid clipped to
  // the head circle, rendered as a Plotly heatmap + electrode dots overlay.

  function ensureTopoGrid() {
    if (state.topoGrid) return state.topoGrid;
    const xy = state.data.electrodes.xy;
    let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;
    for (const [x, y] of xy) {
      if (x < xMin) xMin = x; if (x > xMax) xMax = x;
      if (y < yMin) yMin = y; if (y > yMax) yMax = y;
    }
    const pad = 0.1;
    const cx = 0.5 * (xMin + xMax), cy = 0.5 * (yMin + yMax);
    const radius = Math.max(xMax - cx, yMax - cy) * (1 + pad);
    const N = 64;
    const xs = new Array(N), ys = new Array(N);
    for (let i = 0; i < N; i++) {
      xs[i] = cx - radius + 2 * radius * (i / (N - 1));
      ys[i] = cy - radius + 2 * radius * (i / (N - 1));
    }
    // Pre-compute IDW weights per grid cell over all electrodes.
    const w = new Float32Array(N * N * xy.length);
    const inside = new Uint8Array(N * N);
    const r2 = radius * radius;
    for (let iy = 0; iy < N; iy++) {
      for (let ix = 0; ix < N; ix++) {
        const dx = xs[ix] - cx, dy = ys[iy] - cy;
        const cellInside = (dx*dx + dy*dy) <= r2;
        inside[iy * N + ix] = cellInside ? 1 : 0;
        if (!cellInside) continue;
        let sumW = 0;
        const base = (iy * N + ix) * xy.length;
        for (let e = 0; e < xy.length; e++) {
          const ex = xs[ix] - xy[e][0], ey = ys[iy] - xy[e][1];
          const d2 = ex*ex + ey*ey;
          const wi = 1.0 / (d2 + 1e-3);   // IDW with small epsilon
          w[base + e] = wi;
          sumW += wi;
        }
        for (let e = 0; e < xy.length; e++) w[base + e] /= sumW;
      }
    }
    state.topoGrid = { xs, ys, w, inside, N, cx, cy, radius, nElec: xy.length };
    return state.topoGrid;
  }

  function interpolateField(values) {
    const G = ensureTopoGrid();
    const z = new Array(G.N);
    for (let iy = 0; iy < G.N; iy++) {
      const row = new Array(G.N);
      for (let ix = 0; ix < G.N; ix++) {
        if (!G.inside[iy * G.N + ix]) { row[ix] = null; continue; }
        let s = 0;
        const base = (iy * G.N + ix) * G.nElec;
        for (let e = 0; e < G.nElec; e++) s += G.w[base + e] * values[e];
        row[ix] = s;
      }
      z[iy] = row;
    }
    return z;
  }

  // p99 of |signal| computed separately per series so a near-flat reconstruction
  // doesn't get washed out by a much larger ground truth scale.
  function computeSignalScales() {
    const flat = which => {
      const arr = state.data[which];
      const out = [];
      for (const ch of arr) for (const v of ch) out.push(Math.abs(v));
      out.sort((a, b) => a - b);
      const p = out[Math.floor(0.99 * out.length)] || 1;
      return p * state.data.signal_scale;
    };
    state.signalScale = {
      hd_truth: flat("hd_truth"),
      hd_pred:  flat("hd_pred"),
    };
  }

  function topomapTraces(values, which) {
    const G = ensureTopoGrid();
    const cMax = state.signalScale[which] || 1;
    const z = interpolateField(values);
    const heat = {
      type: "heatmap",
      x: G.xs, y: G.ys, z,
      zmin: -cMax, zmax: cMax,
      colorscale: "RdBu", reversescale: true,
      showscale: false, hoverinfo: "skip",
    };
    const dots = {
      type: "scatter", mode: "markers",
      x: state.data.electrodes.xy.map(p => p[0]),
      y: state.data.electrodes.xy.map(p => p[1]),
      marker: { size: 5, color: "#0e1117", opacity: 0.7 },
      hoverinfo: "skip", showlegend: false,
    };
    // Outline of the head circle
    const N = 96, ring_x = new Array(N), ring_y = new Array(N);
    for (let i = 0; i < N; i++) {
      const a = 2 * Math.PI * i / (N - 1);
      ring_x[i] = G.cx + G.radius * Math.cos(a);
      ring_y[i] = G.cy + G.radius * Math.sin(a);
    }
    const ring = {
      type: "scatter", mode: "lines",
      x: ring_x, y: ring_y,
      line: { color: "#9da7b3", width: 1.2 },
      hoverinfo: "skip", showlegend: false,
    };
    return [heat, ring, dots];
  }

  function redraw(fullRelayout) {
    if (!state.data) return;
    if (!state.signalScale) computeSignalScales();

    const t3 = build3DTraces();
    const layout3d = {
      ...layoutCommon,
      uirevision: "static-3d",   // PRESERVE camera & zoom across updates
      scene: {
        xaxis: { title: "x (mm)", color: "#9da7b3", gridcolor: "#30363d" },
        yaxis: { title: "y (mm)", color: "#9da7b3", gridcolor: "#30363d" },
        zaxis: { title: "z (mm)", color: "#9da7b3", gridcolor: "#30363d" },
        bgcolor: "#161b22",
        aspectmode: "data",
        dragmode: "orbit",
      },
    };

    const tracesGT   = topomapTraces(signalAt(state.t, "hd_truth"), "hd_truth");
    const tracesPred = topomapTraces(signalAt(state.t, "hd_pred"),  "hd_pred");
    const layout2d = {
      ...layoutCommon,
      uirevision: "static-2d",
      xaxis: { visible: false, scaleanchor: "y" },
      yaxis: { visible: false },
      margin: { l: 8, r: 8, t: 8, b: 8 },
    };

    Plotly.react("plot3d",    t3.traces,   layout3d, { responsive: true, displayModeBar: false });
    Plotly.react("plot-gt",   tracesGT,    layout2d, { responsive: true, displayModeBar: false });
    Plotly.react("plot-pred", tracesPred,  layout2d, { responsive: true, displayModeBar: false });

    const tSample = state.data.time_axis[state.t];
    const sec = (tSample / state.data.sampling_rate).toFixed(3);
    $("t-label").textContent =
      `${state.t} / ${state.data.n_time - 1}  (sample ${tSample}, ${sec}s)`;
    $("diag").textContent =
      `Top ${t3.groupCount} of ${state.keptGroups.length} mixture points · ` +
      `${t3.gaussCount} ellipsoids (G=${state.data.n_gaussians_per_point || 1}) · ` +
      `colour fixed to ±max group |amp| = ${state.maxGroupAmpAbs.toFixed(3)} · ` +
      `GT scale ±${state.signalScale.hd_truth.toFixed(3)} · ` +
      `EMAG scale ±${state.signalScale.hd_pred.toFixed(3)}`;
  }

  // --- play / pause ---------------------------------------------------------

  function play() {
    if (state.playing || !state.data) return;
    state.playing = true;
    $("play").textContent = "⏸  Pause";
    const ms = Math.max(1, parseInt($("speed").value, 10));
    state.intervalId = setInterval(() => {
      state.t = (state.t + 1) % state.data.n_time;
      $("t").value = state.t;
      redraw(false);
    }, ms);
  }

  function pause() {
    state.playing = false;
    $("play").textContent = "▶  Play";
    if (state.intervalId) { clearInterval(state.intervalId); state.intervalId = null; }
  }

  // --- init -----------------------------------------------------------------

  async function init() {
    const items = await loadIndex();
    if (!items.length) {
      $("meta").textContent = "No pre-rendered data found in site/data/. Run scripts/build_site_data.py.";
      return;
    }
    const sel = $("source");
    items.forEach(it => {
      const o = document.createElement("option");
      o.value = it.file; o.textContent = it.label;
      sel.appendChild(o);
    });

    sel.addEventListener("change", () => loadData(sel.value));
    $("topk").addEventListener("input", () => {
      const v = parseInt($("topk").value, 10);
      if (!isNaN(v) && v > 0) { state.topk = v; redraw(false); }
    });
    $("t").addEventListener("input", () => { state.t = parseInt($("t").value, 10); redraw(false); });
    $("play").addEventListener("click", () => state.playing ? pause() : play());
    $("reset").addEventListener("click", () => { pause(); state.t = 0; $("t").value = 0; redraw(false); });
    $("speed").addEventListener("input", () => { if (state.playing) { pause(); play(); } });

    const clip = $("clip"); if (clip) clip.addEventListener("change", () => {
      state.clipToHead = clip.checked; recomputeKept(); redraw(true);
    });
    const sph = $("sphere"); if (sph) sph.addEventListener("change", () => {
      state.showSphere = sph.checked; redraw(true);
    });
    const sig = $("sigma"); if (sig) sig.addEventListener("input", () => {
      const v = parseFloat(sig.value);
      if (!isNaN(v) && v > 0) { state.sigmaScale = v; recomputeKept(); redraw(true); }
    });
    const op = $("ellopacity"); if (op) op.addEventListener("input", () => {
      const v = parseFloat(op.value);
      if (!isNaN(v) && v > 0) { state.ellipsoidOpacity = v; redraw(true); }
    });

    state.topk = parseInt($("topk").value, 10);
    if (clip) state.clipToHead = clip.checked;
    if (sph) state.showSphere = sph.checked;
    await loadData(items[0].file);
  }

  init().catch(err => {
    console.error(err);
    $("meta").textContent = `Error: ${err.message}`;
  });
})();
