// static/marketdata/js/dashboard.js
// Market Data dashboard wired to TradeIntel API (snapshots/ohlc/line), with resize-safe charts.

(function (global) {
  "use strict";

  const S = {
    endpoints: {},
    defaults: { symbol: "AAPL", timeframe: "D1", autoRefreshSec: 30 },
    refreshTimer: null,
    symbols: [],
  };

  // ---- utils ----
  const byId = (id) => document.getElementById(id);
  const setText = (id, val) => { const el = byId(id); if (el) el.textContent = val; };
  const fmtNum = (n, dp = 2) => (n ?? 0).toLocaleString(undefined, { maximumFractionDigits: dp });
  const dateToISO = (d) => `${d.getFullYear()}-${String(d.getMonth()+1).padStart(2,"0")}-${String(d.getDate()).padStart(2,"0")}`;

  function applyPreset(preset) {
    const startEl = byId("startDate"), endEl = byId("endDate");
    if (!startEl || !endEl) return;
    const today = new Date(); let start = null; const end = new Date();
    switch (preset) {
      case "7d": start = new Date(); start.setDate(today.getDate()-7); break;
      case "1m": start = new Date(); start.setMonth(today.getMonth()-1); break;
      case "3m": start = new Date(); start.setMonth(today.getMonth()-3); break;
      case "ytd": start = new Date(today.getFullYear(),0,1); break;
      case "1y": start = new Date(); start.setFullYear(today.getFullYear()-1); break;
      case "custom": default: startEl.disabled = false; endEl.disabled = false; return;
    }
    startEl.disabled = true; endEl.disabled = true;
    startEl.value = dateToISO(start); endEl.value = dateToISO(end);
  }

  function getSelectedRange() {
    const preset = byId("presetSelect")?.value || "1m";
    if (preset === "custom") {
      return { start: byId("startDate")?.value || null, end: byId("endDate")?.value || null };
    }
    return { start: byId("startDate")?.value || null, end: byId("endDate")?.value || null };
  }

  function updateTimestamp() { setText("lastUpdated", new Date().toLocaleString()); }

  // ---- fetch helpers ----
  async function fetchJSON(url) {
    console.log("[API] GET", url);
    const r = await fetch(url, { headers: { "X-Requested-With": "XMLHttpRequest" } });
    if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
    return r.json();
  }

  // ---- parsers tuned to your API ----
  // line: {symbol, data:[{timestamp,price}]}
  function parseLineResponse(resp) {
    const rows = Array.isArray(resp) ? resp : (resp?.data || []);
    return {
      xs: rows.map(r => r.timestamp || r.ts || r.time || r.date),
      ys: rows.map(r => r.price ?? r.close ?? r.value),
    };
  }

  // ohlc: {symbol, ohlc:[{timestamp, open, high, low, close, volume}]}
  function parseOHLCResponse(resp) {
    const rows = resp?.ohlc || (Array.isArray(resp) ? resp : (resp?.results || resp?.data || []));
    return rows.map(r => ({
      x: r.timestamp || r.ts || r.time || r.date,
      open: r.open, high: r.high, low: r.low, close: r.close,
      volume: r.volume || 0, source: r.source, symbol: resp?.symbol,
    }));
  }

  // ---- chart resize handling ----
  function resizeCharts() {
    if (!global.Plotly) return;
    ["lineChart", "candleChart"].forEach(id => {
      const el = byId(id);
      if (el) global.Plotly.Plots.resize(el);
    });
  }

  // ---- data loaders ----
  async function loadSymbols() {
    try {
      const data = await fetchJSON(S.endpoints.prices);
      const list = data?.snapshots || data?.results || data?.data || (Array.isArray(data) ? data : []);
      const set = new Set();
      list.forEach(r => r?.symbol && set.add(r.symbol));
      if (!set.size) ["AAPL", "MSFT", "US100", "BTC/USD"].forEach(s => set.add(s));
      S.symbols = [...set].sort();

      // wire UI
      const dl = byId("symbolList");
      if (dl) dl.innerHTML = S.symbols.map(s => `<option value="${s}">`).join("");
      const liveSel = byId("filterSymbolLive");
      if (liveSel) liveSel.innerHTML = `<option value="">All Symbols</option>` + S.symbols.map(s => `<option>${s}</option>`).join("");
      const ohlcSel = byId("filterSymbolOHLC");
      if (ohlcSel) ohlcSel.innerHTML = `<option value="">All Symbols</option>` + S.symbols.map(s => `<option>${s}</option>`).join("");

      const input = byId("symbolInput");
      if (input && !input.value) input.value = S.defaults.symbol || S.symbols[0] || "AAPL";
      console.log("[UI] symbols:", S.symbols.length, S.symbols.slice(0, 5));
    } catch (e) {
      console.warn("loadSymbols failed:", e);
      S.symbols = ["AAPL", "MSFT", "US100", "BTC/USD"];
    }
  }

  async function loadLineChart(symbol, start, end) {
    if (!global.Plotly) { console.warn("Plotly not loaded"); return; }
    let url = `${S.endpoints.line}?symbol=${encodeURIComponent(symbol)}`;
    // your line endpoint accepts start/end (same names)
    if (start) url += `&start=${encodeURIComponent(start)}`;
    if (end) url += `&end=${encodeURIComponent(end)}`;
    const resp = await fetchJSON(url);
    const { xs, ys } = parseLineResponse(resp);
    console.log("[Chart] line points:", xs.length);

    const trace = { x: xs, y: ys, type: "scatter", mode: "lines", name: symbol };
    const layout = { margin: { t: 10, r: 20, b: 30, l: 45 }, xaxis: { type: "date" } };
    global.Plotly.react("lineChart", [trace], layout, { responsive: true, displaylogo: false });
    setTimeout(resizeCharts, 0);
  }

  // Try multiple timeframe options if needed, using ?tf= (your API)
  const TF_CANDIDATES = ["D1", "1D", "1d", "D", "DAY", null];

  async function tryFetchOHLC(symbol, start, end, tf) {
    let url = `${S.endpoints.ohlc}?symbol=${encodeURIComponent(symbol)}`;
    if (tf) url += `&tf=${encodeURIComponent(tf)}`; // <-- your API expects tf=
    if (start) url += `&start=${encodeURIComponent(start)}`;
    if (end) url += `&end=${encodeURIComponent(end)}`;
    const resp = await fetchJSON(url);
    const rows = parseOHLCResponse(resp);
    console.log(`[OHLC] fetched ${rows.length} rows with tf=`, tf ?? "(none)");
    return rows;
  }

  async function loadCandleChart(symbol, start, end, timeframe) {
    if (!global.Plotly) { console.warn("Plotly not loaded"); return; }

    const attempts = [timeframe, ...TF_CANDIDATES.filter(tf => tf !== timeframe)];
    let rows = [];
    for (const tf of attempts) {
      try {
        rows = await tryFetchOHLC(symbol, start, end, tf);
        if (rows.length) break;
      } catch (e) {
        console.warn("OHLC fetch failed for tf:", tf, e);
      }
    }

    const x = rows.map(r => r.x);
    const open = rows.map(r => r.open);
    const high = rows.map(r => r.high);
    const low  = rows.map(r => r.low);
    const close= rows.map(r => r.close);
    const volume = rows.map(r => r.volume);

    const data = [];
    if (x.length) {
      data.push({ x, open, high, low, close, type: "candlestick", name: symbol });
      data.push({ x, y: volume, type: "bar", name: "Volume", yaxis: "y2", opacity: 0.4 });
    }
    const layout = {
      margin: { t: 10, r: 20, b: 30, l: 45 },
      xaxis: { type: "date" },
      yaxis: { title: "Price" },
      yaxis2: { overlaying: "y", side: "right", showgrid: false, rangemode: "tozero" },
      annotations: x.length ? [] : [{ text: "No OHLC data", xref: "paper", yref: "paper", x: .5, y: .5, showarrow: false }]
    };
    console.log("[Chart] candle bars:", x.length);
    global.Plotly.react("candleChart", data, layout, { responsive: true, displaylogo: false });
    setTimeout(resizeCharts, 0);
  }

  async function loadLivePricesTable() {
    const data = await fetchJSON(S.endpoints.prices);
    const list = data?.snapshots || data?.results || data?.data || (Array.isArray(data) ? data : []);
    console.log("[Live] rows:", list.length);

    const tbody = byId("liveTable")?.querySelector("tbody");
    if (!tbody) return;
    const filter = byId("filterSymbolLive")?.value || "";
    const rows = list.filter(r => !filter || r.symbol === filter);

    if (!rows.length) {
      tbody.innerHTML = `<tr><td colspan="5" class="text-muted small">No live prices yet.</td></tr>`;
    } else {
      tbody.innerHTML = rows.map(r => {
        const pct = r.pct_change ?? r.change_pct ?? null;
        const pctHTML = pct == null ? "—" : `<span class="${pct >= 0 ? 'text-success' : 'text-danger'}">${(pct ?? 0).toFixed(2)}%</span>`;
        return `
          <tr>
            <td class="fw-semibold">${r.symbol}</td>
            <td class="text-end">${fmtNum(r.price ?? r.last ?? 0, 6)}</td>
            <td class="text-end">${pctHTML}</td>
            <td class="text-nowrap">${r.timestamp ?? r.ts ?? r.time ?? ""}</td>
            <td>${r.source ?? ""}</td>
          </tr>`;
      }).join("");
    }

    const uniq = new Set(rows.map(r => r.symbol));
    setText("kpiActiveSymbols", String(uniq.size || 0));
  }

  async function loadOHLCTable() {
    const symbol = byId("filterSymbolOHLC")?.value || byId("symbolInput")?.value || S.symbols[0] || S.defaults.symbol;
    const timeframe = byId("timeframeSelect")?.value || S.defaults.timeframe;

    const attempts = [timeframe, ...TF_CANDIDATES.filter(tf => tf !== timeframe)];
    let rows = [];
    for (const tf of attempts) {
      try {
        rows = await tryFetchOHLC(symbol, null, null, tf);
        if (rows.length) break;
      } catch (e) {
        console.warn("OHLC table fetch failed tf:", tf, e);
      }
    }

    const tbody = byId("ohlcTable")?.querySelector("tbody");
    if (!tbody) return;

    if (!rows.length) {
      tbody.innerHTML = `<tr><td colspan="8" class="text-muted small">No OHLC rows found.</td></tr>`;
    } else {
      tbody.innerHTML = rows.slice(-100).reverse().map(r => `
        <tr>
          <td>${r.symbol ?? symbol}</td>
          <td>${r.x || ""}</td>
          <td class="text-end">${fmtNum(r.open, 6)}</td>
          <td class="text-end">${fmtNum(r.high, 6)}</td>
          <td class="text-end">${fmtNum(r.low, 6)}</td>
          <td class="text-end">${fmtNum(r.close, 6)}</td>
          <td class="text-end">${fmtNum(r.volume ?? 0)}</td>
          <td>${r.source ?? ""}</td>
        </tr>
      `).join("");
    }
  }

  // ---- orchestrator ----
  async function refreshAll({ charts = true, live = true, ohlc = true } = {}) {
    updateTimestamp();
    if (!S.symbols.length) await loadSymbols();

    const input = byId("symbolInput");
    if (input && !input.value) input.value = S.symbols[0] || S.defaults.symbol;

    const symbol = input?.value || S.symbols[0] || S.defaults.symbol;
    const timeframe = byId("timeframeSelect")?.value || S.defaults.timeframe;
    const { start, end } = getSelectedRange();

    const tasks = [];
    if (charts) { tasks.push(loadLineChart(symbol, start, end)); tasks.push(loadCandleChart(symbol, start, end, timeframe)); }
    if (live) tasks.push(loadLivePricesTable());
    if (ohlc) tasks.push(loadOHLCTable());
    await Promise.allSettled(tasks);
  }

  function startAutoRefresh() {
    clearInterval(S.refreshTimer);
    const tgl = byId("autoRefreshToggle");
    if (tgl && tgl.checked) {
      const ms = (S.defaults.autoRefreshSec || 30) * 1000;
      S.refreshTimer = setInterval(() => refreshAll({ charts: false }), ms);
    }
  }

  // ---- events ----
  function wireEvents() {
    byId("presetSelect")?.addEventListener("change", (e) => applyPreset(e.target.value));
    byId("btnApply")?.addEventListener("click", () => refreshAll());
    byId("btnRefreshNow")?.addEventListener("click", () => refreshAll({}));
    byId("autoRefreshToggle")?.addEventListener("change", startAutoRefresh);
    byId("filterSymbolLive")?.addEventListener("change", () => refreshAll({ charts: false, ohlc: false }));
    byId("filterSymbolOHLC")?.addEventListener("change", () => refreshAll({ charts: false, live: false }));
    byId("btnTestSound")?.addEventListener("click", () => byId("beepUp")?.play().catch(()=>{}));

    // Resize charts when switching tabs (Bootstrap 5)
    document.querySelectorAll('button[data-bs-toggle="tab"]').forEach(btn => {
      btn.addEventListener('shown.bs.tab', () => resizeCharts());
    });

    // Window resize
    global.addEventListener('resize', () => resizeCharts());
  }

  // ---- export ----
  global.MarketDataDashboard = {
    init(config) {
      S.endpoints = config?.endpoints || S.endpoints;
      S.defaults = Object.assign(S.defaults, config?.defaults || {});
      wireEvents();
      applyPreset(byId("presetSelect")?.value || "1m");
      refreshAll().then(startAutoRefresh).catch((e) => console.error("Initial refresh failed:", e));
    }
  };
})(window);
