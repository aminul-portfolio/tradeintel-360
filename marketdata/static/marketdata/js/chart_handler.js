// ✅ Register Chart.js plugin
if (typeof Chart !== "undefined" && typeof ChartDataLabels !== "undefined") {
    Chart.register(ChartDataLabels);
}

// ✅ Global chart instances
let comparisonChart = null;
let trendChart = null;
window.customTrendChart = null;

// ✅ Comparison Chart: Current vs Yesterday
window.renderPriceComparison = function (quote, yesterday_price, current_price, diff, diff_percent) {
    const el = document.getElementById("compareChart");
    if (!el || typeof Chart === "undefined") return;
    if (comparisonChart instanceof Chart) comparisonChart.destroy();

    comparisonChart = new Chart(el, {
        type: "bar",
        data: {
            labels: ["Yesterday", "Now"],
            datasets: [{
                label: `${quote} Price`,
                data: [yesterday_price, current_price],
                backgroundColor: [
                    diff >= 0 ? "#28a745" : "#dc3545",
                    "#007bff"
                ],
                borderRadius: 6
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: { display: false },
                datalabels: {
                    color: '#000',
                    anchor: 'end',
                    align: 'top',
                    font: { weight: 'bold' },
                    formatter: (value, ctx) =>
                        ctx.dataIndex === 1 ? `${diff >= 0 ? '+' : ''}${diff_percent}%` : `$${value}`
                },
                tooltip: {
                    callbacks: {
                        label: ctx => `$${ctx.raw}`
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    ticks: { callback: v => `$${v}` }
                }
            }
        },
        plugins: [ChartDataLabels]
    });
};

// ✅ 7-Day Trend Chart
window.renderWeeklyTrend = function (quote, trend_labels, trend_prices) {
    const el = document.getElementById("trendChart");
    if (!el || typeof Chart === "undefined") return;
    if (trendChart instanceof Chart) trendChart.destroy();

    trendChart = new Chart(el, {
        type: "line",
        data: {
            labels: trend_labels,
            datasets: [{
                label: `${quote} (Trend)`,
                data: trend_prices,
                borderColor: "#007bff",
                backgroundColor: "rgba(0,123,255,0.1)",
                fill: true,
                tension: 0.3,
                pointRadius: 5,
                pointHoverRadius: 8,
                pointBorderWidth: 1,
                pointHoverBorderWidth: 3
            }]
        },
        options: {
            responsive: true,
            plugins: {
                tooltip: {
                    callbacks: { label: ctx => `$${ctx.raw}` }
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    ticks: { callback: v => `$${v}` }
                }
            }
        }
    });
};

// ✅ Download any chart
window.downloadChart = function (canvasId, filename) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const link = document.createElement("a");
    link.href = canvas.toDataURL("image/png");
    link.download = `${filename}.png`;
    link.click();
};

// ✅ Load custom trend range chart (Crypto + Index support)
window.loadCustomRange = function () {
    const from = document.getElementById("dateFrom").value;
    const to = document.getElementById("dateTo").value;

    if (!from || !to || typeof quote === "undefined") {
        alert("Please select both start and end dates.");
        return;
    }

    // 🔄 Smart endpoint for crypto vs index
    const isIndex = ["US100", "SPX", "GOLD"].includes(quote.toUpperCase());
    const apiUrl = isIndex
        ? `/api/custom-trend/${quote.toLowerCase()}/?start=${from}&end=${to}`
        : `/api/custom-trend/?quote=${quote}&start=${from}&end=${to}`;

    fetch(apiUrl)
        .then(res => res.json())
        .then(data => {
            if (data.error) return alert(data.error);
            const ctx = document.getElementById("customTrendChart");
            if (!ctx) return;

            if (window.customTrendChart instanceof Chart) {
                window.customTrendChart.destroy();
                window.customTrendChart = null;
            }

            window.customTrendChart = new Chart(ctx, {
                type: "line",
                data: {
                    labels: data.labels,
                    datasets: [{
                        label: `${quote} Custom Trend`,
                        data: data.prices,
                        borderColor: "#6f42c1",
                        backgroundColor: "rgba(111, 66, 193, 0.2)",
                        fill: true,
                        tension: 0.3
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        tooltip: {
                            callbacks: {
                                label: ctx => `$${ctx.raw}`
                            }
                        }
                    },
                    scales: {
                        y: {
                            ticks: { callback: v => `$${v}` }
                        }
                    }
                }
            });

            document.getElementById("customChangeText").innerText =
                `${data.change >= 0 ? '+' : ''}$${data.change} (${data.change_percent.toFixed(2)}%)`;
        })
        .catch(err => {
            alert("Error loading trend data.");
            console.error(err);
        });
};
