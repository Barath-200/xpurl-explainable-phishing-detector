const API = 'http://localhost:5000/api';

// ── Tab switching ──
function switchTab(name) {
  document.querySelectorAll('.tab-btn').forEach((b, i) => {
    b.classList.toggle('active', b.textContent.toLowerCase().startsWith(name.charAt(0) === 's' ? 'single' : name.charAt(0) === 'b' ? 'batch' : 'about'));
  });
  document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
  document.getElementById('panel-' + name).classList.add('active');
}

document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
  });
});

// ── Set example URL ──
function setExample(el) {
  document.getElementById('url-input').value = el.textContent.trim();
  document.getElementById('url-input').focus();
}

// ── Update batch count ──
function updateCount() {
  const lines = document.getElementById('batch-input').value
    .split('\n').map(s => s.trim()).filter(Boolean);
  document.getElementById('batch-count').textContent = lines.length + ' URL' + (lines.length !== 1 ? 's' : '');
}

// ── Enter key on single input ──
document.getElementById('url-input').addEventListener('keydown', e => {
  if (e.key === 'Enter') runScan();
});

// ── Single scan ──
async function runScan() {
  const url = document.getElementById('url-input').value.trim();
  const errDiv = document.getElementById('error-msg');
  errDiv.style.display = 'none';

  if (!url) { showError(errDiv, 'Please enter a URL to analyse.'); return; }

  const btn = document.getElementById('scan-btn');
  btn.disabled = true;
  btn.classList.add('loading');

  try {
    const res  = await fetch(API + '/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url }),
    });
    const data = await res.json();
    if (data.error) { showError(errDiv, data.error); return; }
    renderResult(data);
  } catch (err) {
    showError(errDiv, 'Cannot reach API server. Make sure app.py is running on localhost:5000.');
  } finally {
    btn.disabled = false;
    btn.classList.remove('loading');
  }
}

// ── Batch scan ──
async function runBatch() {
  const raw = document.getElementById('batch-input').value;
  const urls = raw.split('\n').map(s => s.trim()).filter(Boolean);
  const errDiv = document.getElementById('batch-error');
  errDiv.style.display = 'none';

  if (!urls.length) { showError(errDiv, 'Please enter at least one URL.'); return; }
  if (urls.length > 50) { showError(errDiv, 'Maximum 50 URLs per batch.'); return; }

  const btn = document.getElementById('batch-btn');
  btn.disabled = true;
  btn.classList.add('loading');

  try {
    const res  = await fetch(API + '/predict_batch', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ urls }),
    });
    const data = await res.json();
    if (data.error) { showError(errDiv, data.error); return; }
    renderBatch(data.results);
  } catch (err) {
    showError(errDiv, 'Cannot reach API server. Make sure app.py is running on localhost:5000.');
  } finally {
    btn.disabled = false;
    btn.classList.remove('loading');
  }
}

// ── Show error ──
function showError(el, msg) {
  el.textContent = '⚠ ' + msg;
  el.style.display = 'block';
}

// ── Render single result ──
function renderResult(d) {
  const isPhish  = d.label === 'PHISHING';
  const riskCls  = d.risk.toLowerCase();
  const probPct  = (d.probability * 100).toFixed(1) + '%';
  const threshPct = (d.threshold * 100).toFixed(1);

  // SHAP bars
  const maxAbs = Math.max(...(d.shap_drivers || []).map(x => Math.abs(x.value)), 0.001);
  const shapRows = (d.shap_drivers || []).map(s => {
    const pct = (Math.abs(s.value) / maxAbs * 48).toFixed(1);
    const dir = s.value > 0 ? 'positive' : 'negative';
    const sign = s.value > 0 ? '+' : '';
    return `
      <div class="shap-bar-row">
        <div class="shap-name">${esc(s.feature)}</div>
        <div class="shap-track">
          <div class="shap-zero"></div>
          <div class="shap-fill ${dir}" style="width:${pct}%"></div>
        </div>
        <div class="shap-value">${sign}${s.value.toFixed(4)}</div>
      </div>`;
  }).join('');

  // Feature cells
  const featCells = Object.entries(d.features || {}).map(([k, v]) => {
    const flagged = (k === 'brand_impersonation' || k === 'has_ip' || k === 'sus_tld' || k === 'has_sus_keywords') && v > 0;
    return `
      <div class="feature-cell">
        <span class="feat-name">${esc(k)}</span>
        <span class="feat-val${flagged ? ' flagged' : ''}">${typeof v === 'number' ? v.toFixed(v % 1 === 0 ? 0 : 3) : v}</span>
      </div>`;
  }).join('');

  const modeBadge = d.mode === 'demo'
    ? '<span class="mode-badge demo">demo mode — heuristic scoring</span>'
    : '<span class="mode-badge model">model inference</span>';

  document.getElementById('result-area').innerHTML = `
    <div class="result-card">

      <div class="verdict-bar ${isPhish ? 'phishing' : 'legitimate'}">
        <div class="verdict-icon">${isPhish ? '⚠' : '✓'}</div>
        <div class="verdict-text">
          <h2>${isPhish ? 'Phishing Detected' : 'Legitimate URL'}</h2>
          <div class="verdict-url">${esc(d.url)}</div>
          ${modeBadge}
        </div>
      </div>

      <div class="metrics-row">
        <div class="metric-cell">
          <div class="metric-label">Phishing Prob</div>
          <div class="metric-value">${probPct}</div>
        </div>
        <div class="metric-cell">
          <div class="metric-label">Risk Level</div>
          <div class="metric-value risk-${riskCls}">${d.risk}</div>
        </div>
        <div class="metric-cell">
          <div class="metric-label">Threshold</div>
          <div class="metric-value" style="font-size:18px">${d.threshold}</div>
        </div>
      </div>

      <div class="prob-section">
        <div class="prob-label">
          <span>Probability spectrum</span>
          <span>${probPct}</span>
        </div>
        <div class="prob-track">
          <div class="prob-fill ${riskCls}" id="prob-fill" style="width:0%"></div>
          <div class="prob-threshold" style="left:${threshPct}%" data-label="thresh ${d.threshold}"></div>
        </div>
      </div>

      ${shapRows ? `
      <div class="shap-section">
        <div class="section-title">SHAP feature contributions</div>
        ${shapRows}
        <div class="shap-legend">
          <div class="legend-item"><div class="legend-dot" style="background:rgba(239,68,68,0.7)"></div>Towards phishing</div>
          <div class="legend-item"><div class="legend-dot" style="background:rgba(59,130,246,0.7)"></div>Towards legitimate</div>
        </div>
      </div>` : ''}

      ${featCells ? `
      <div class="features-section">
        <button class="features-toggle" id="feat-toggle" onclick="toggleFeatures()">
          <span class="arrow">▶</span> Show raw features
        </button>
        <div class="features-grid" id="feat-grid">${featCells}</div>
      </div>` : ''}

    </div>`;

  // Animate probability bar
  setTimeout(() => {
    const fill = document.getElementById('prob-fill');
    if (fill) fill.style.width = (d.probability * 100).toFixed(1) + '%';
  }, 60);
}

function toggleFeatures() {
  const grid   = document.getElementById('feat-grid');
  const toggle = document.getElementById('feat-toggle');
  const open   = grid.classList.toggle('visible');
  toggle.classList.toggle('open', open);
  toggle.querySelector('.arrow').textContent = open ? '▼' : '▶';
  toggle.childNodes[1].textContent = (open ? ' Hide' : ' Show') + ' raw features';
}

// ── Render batch ──
function renderBatch(results) {
  if (!results || !results.length) {
    document.getElementById('batch-results').innerHTML = '<p style="color:var(--text3);font-size:13px;margin-top:12px;">No results.</p>';
    return;
  }

  const phishCount = results.filter(r => r.label === 'PHISHING').length;
  const rows = results.map(r => {
    const isP = r.label === 'PHISHING';
    return `
      <div class="batch-row">
        <span class="batch-url">${esc(r.url)}</span>
        <span class="badge ${isP ? 'badge-phishing' : 'badge-legitimate'}">${r.label}</span>
        <span class="badge badge-${r.risk.toLowerCase()}">${r.risk}</span>
        <span class="prob-pill">${(r.probability * 100).toFixed(1)}%</span>
      </div>`;
  }).join('');

  document.getElementById('batch-results').innerHTML = `
    <div class="batch-table-wrap">
      <div class="batch-table-header">
        <span class="batch-title">Scan results</span>
        <div class="batch-summary">
          <span style="color:var(--red)">${phishCount} phishing</span>
          <span style="color:var(--green)">${results.length - phishCount} legitimate</span>
        </div>
      </div>
      ${rows}
    </div>`;
}

// ── Sanitise ──
function esc(s) {
  return String(s)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')
    .replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}

// ── Load health info ──
async function loadHealth() {
  try {
    const res  = await fetch(API + '/health');
    const data = await res.json();
    document.querySelector('.header-meta').innerHTML = `
      <div>Model: ${data.model}</div>
      <div>Device: ${data.device}</div>`;

    const m = data.metadata || {};
    document.getElementById('meta-f1').textContent     = m.val_f1_ewc    || m.val_f1    || '—';
    document.getElementById('meta-auc').textContent    = m.val_auc_ewc   || m.val_auc   || '—';
    document.getElementById('meta-thresh').textContent = data.threshold  || '—';
    document.getElementById('meta-device').textContent = data.device     || '—';
    document.getElementById('meta-mode').textContent   = data.model      || '—';
  } catch {
    document.querySelector('.header-meta').innerHTML = `
      <div style="color:var(--amber)"><span class="dot" style="background:var(--amber)"></span>API offline</div>
      <div>Start app.py to connect</div>`;
  }
}

loadHealth();