document.addEventListener("DOMContentLoaded", function () {
  if (typeof quote === "undefined" || !quote) return;

  // refresh every 60s
  setInterval(() => {
    fetch(`/api/live-price-api/?quote=${encodeURIComponent(quote)}`)
      .then(response => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.json();
      })
      .then(data => {
        if (data.error) return;

        // Update the "Current vs Yesterday" chart
        if (typeof renderPriceComparison === "function") {
          renderPriceComparison(
            quote,
            data.yesterday_price,
            data.current_price,
            data.diff,
            data.diff_percent
          );
        }

        // (Optional) update the summary list in the DOM
        const summary = document.querySelector(".list-group");
        if (summary) {
          summary.innerHTML = `
            <li class="list-group-item"><strong>Current Price:</strong> $${data.current_price.toFixed(2)}</li>
            <li class="list-group-item"><strong>Yesterday's Close:</strong> $${data.yesterday_price.toFixed(2)}</li>
            <li class="list-group-item"><strong>Difference:</strong>
              ${
                data.diff >= 0
                  ? `<span class="text-success">+$${data.diff} (+${data.diff_percent}%)</span>`
                  : `<span class="text-danger">$${data.diff} (${data.diff_percent}%)</span>`
              }
            </li>
          `;
        }
      })
      .catch(err => console.error("Live price refresh error:", err));
  }, 60000);
});
