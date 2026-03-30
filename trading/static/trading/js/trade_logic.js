/* ═══════════════════════════════════════════════════════════════════
   TRADE LOGIC — trading/static/trading/js/trade_logic.js
   Live R/R preview on trade_form.html.
   ═══════════════════════════════════════════════════════════════════ */

(function () {
  'use strict';

  // ── R/R quality thresholds ──────────────────────────────────────
  // Controls which CSS class (and colour) is applied to #rrPreview.
  // Adjust these to match your trading standards.
  const RR_THRESHOLDS = {
    poor:   0.5,   // below this: red
    fair:   1.0,   // below this: amber
    good:   2.0,   // below this: green
    // at or above good: accent (teal)
  };

  // ── Selectors ───────────────────────────────────────────────────
  const FIELDS = {
    side:   '#id_side',
    entry:  '#id_entry_price',
    sl:     '#id_stop_loss',
    tp:     '#id_take_profit',
  };

  const PREVIEW_ID = 'rrPreview';

  // ── Helpers ─────────────────────────────────────────────────────
  function getVal(selector) {
    const el = document.querySelector(selector);
    if (!el) return null;
    const v = parseFloat(el.value);
    return isNaN(v) ? null : v;
  }

  function getStr(selector) {
    const el = document.querySelector(selector);
    return el ? el.value : null;
  }

  function setPreview(el, rr) {
    // Remove all state classes
    el.classList.remove('rr--empty', 'rr--poor', 'rr--fair', 'rr--good', 'rr--strong');

    if (rr === null) {
      el.textContent = 'Enter side, entry, stop loss, and take profit to see R/R.';
      el.classList.add('rr--empty');
      return;
    }

    if (rr <= 0) {
      el.textContent = 'Invalid setup — reward is zero or negative.';
      el.classList.add('rr--poor');
      return;
    }

    // Quality label
    let quality, stateClass;
    if (rr < RR_THRESHOLDS.poor) {
      quality = 'Poor';
      stateClass = 'rr--poor';
    } else if (rr < RR_THRESHOLDS.fair) {
      quality = 'Below average';
      stateClass = 'rr--poor';
    } else if (rr < RR_THRESHOLDS.good) {
      quality = 'Fair';
      stateClass = 'rr--fair';
    } else if (rr < RR_THRESHOLDS.good * 1.5) {
      quality = 'Good';
      stateClass = 'rr--good';
    } else {
      quality = 'Strong';
      stateClass = 'rr--strong';
    }

    el.textContent = `R/R  ${rr.toFixed(2)}  —  ${quality}`;
    el.classList.add(stateClass);
  }

  // ── Core calculation ────────────────────────────────────────────
  function updateRR() {
    const el = document.getElementById(PREVIEW_ID);
    if (!el) return;

    const side  = getStr(FIELDS.side);
    const entry = getVal(FIELDS.entry);
    const sl    = getVal(FIELDS.sl);
    const tp    = getVal(FIELDS.tp);

    // Need all four values to compute
    if (!side || entry === null || sl === null || tp === null) {
      setPreview(el, null);
      return;
    }

    let risk, reward;

    if (side === 'LONG') {
      risk   = entry - sl;
      reward = tp - entry;
    } else {
      // SHORT
      risk   = sl - entry;
      reward = entry - tp;
    }

    const rr = risk > 0 ? reward / risk : null;
    setPreview(el, rr);
  }

  // ── Initialise ──────────────────────────────────────────────────
  // Listen on all relevant fields; recalculate on any change.
  document.addEventListener('DOMContentLoaded', function () {
    // Initial state
    updateRR();

    // Attach listeners to each relevant field
    Object.values(FIELDS).forEach(function (selector) {
      const el = document.querySelector(selector);
      if (el) {
        el.addEventListener('input',  updateRR);
        el.addEventListener('change', updateRR); // covers <select>
      }
    });
  });

})();