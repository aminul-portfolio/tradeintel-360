// Live RR preview on the form (optional)
document.addEventListener('input', function () {
  const side = document.querySelector('#id_side')?.value;
  const entry = parseFloat(document.querySelector('#id_entry_price')?.value || 0);
  const sl = parseFloat(document.querySelector('#id_stop_loss')?.value || 0);
  const tp = parseFloat(document.querySelector('#id_take_profit')?.value || 0);
  if (!side || !entry || !sl || !tp) return;
  let risk = side === 'LONG' ? (entry - sl) : (sl - entry);
  let reward = side === 'LONG' ? (tp - entry) : (entry - tp);
  const rr = risk > 0 ? (reward / risk).toFixed(2) : '';
  const el = document.getElementById('rrPreview');
  if (el) el.textContent = rr ? `R/R: ${rr}` : '';
});
