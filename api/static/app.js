// ==================== Tab Navigation ====================
document.querySelectorAll('.sidebar li').forEach(tab => {
  tab.addEventListener('click', () => {
    document.querySelectorAll('.sidebar li').forEach(t => t.classList.remove('active'));
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    const tabName = tab.dataset.tab;
    document.getElementById(`tab-${tabName}`).classList.add('active');
    document.getElementById('page-title').textContent = tab.textContent.trim();
    if (tabName === 'overview') loadOverview();
    if (tabName === 'memories') loadMemories();
    if (tabName === 'enterprise') loadEnterprise();
    if (tabName === 'performance') loadPerformance();
    if (tabName === 'monitoring') loadMonitoring();
    if (tabName === 'integrations') loadIntegrations();
  });
});

// ==================== API Helper ====================
async function api(path, method = 'GET', body = null) {
  const opts = { method, headers: {} };
  if (body) {
    opts.headers['Content-Type'] = 'application/json';
    opts.body = JSON.stringify(body);
  }
  const res = await fetch(path, opts);
  return res.json();
}

// ==================== Overview Tab ====================
async function loadOverview() {
  try {
    const stats = await api('/api/dashboard/stats');
    if (stats.success) {
      document.getElementById('stat-vectors').textContent = stats.stats.total_vectors.toLocaleString();
      document.getElementById('stat-collections').textContent = stats.stats.total_collections;
    }
  } catch(e) { console.error('Stats error:', e); }

  try {
    const lat = await api('/api/dashboard/latency');
    if (lat.success) {
      document.getElementById('stat-latency').textContent = lat.latency.avg_ms + 'ms';
    }
  } catch(e) { console.error('Latency error:', e); }

  try {
    const idx = await api('/api/dashboard/index-info');
    if (idx.success) {
      document.getElementById('stat-hnsw').textContent = idx.index_info.hnsw_loaded ? 'Loaded' : 'Not loaded';
    }
  } catch(e) { console.error('Index error:', e); }

  drawLatencyChart();
  drawIndexChart();
}

let _latencyChart = null;
function drawLatencyChart() {
  const ctx = document.getElementById('latency-chart').getContext('2d');
  if (_latencyChart) _latencyChart.destroy();
  _latencyChart = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['Brute Force', 'HNSW', 'IVF'],
      datasets: [{
        label: 'Avg Query Time (ms)',
        data: [42, 3.2, 15],
        backgroundColor: ['#ff6384', '#4fc3f7', '#66bb6a'],
        borderRadius: 6
      }]
    },
    options: {
      responsive: true,
      plugins: { legend: { display: false } },
      scales: { y: { beginAtZero: true, grid: { color: '#f0f0f0' } }, x: { grid: { display: false } } }
    }
  });
}

let _indexChart = null;
function drawIndexChart() {
  const ctx = document.getElementById('index-chart').getContext('2d');
  if (_indexChart) _indexChart.destroy();
  _indexChart = new Chart(ctx, {
    type: 'doughnut',
    data: {
      labels: ['HNSW Nodes', 'IVF Clusters', 'Raw Vectors'],
      datasets: [{
        data: [65, 25, 10],
        backgroundColor: ['#4fc3f7', '#66bb6a', '#ffa726'],
        borderWidth: 0
      }]
    },
    options: {
      responsive: true,
      plugins: { legend: { position: 'bottom' } }
    }
  });
}

// ==================== Search Tab ====================
document.getElementById('search-btn').addEventListener('click', async () => {
  const query = document.getElementById('search-query').value;
  const k = parseInt(document.getElementById('search-k').value);
  const method = document.getElementById('search-method').value;
  if (!query) return;

  const infoBar = document.getElementById('search-info');
  const container = document.getElementById('search-results');
  infoBar.textContent = 'Searching...';
  container.innerHTML = '';

  try {
    const colls = await api('/collections');
    if (colls.success && colls.collections && colls.collections.length > 0) {
      const collId = colls.collections[0].collection_id || colls.collections[0].id;
      const result = await api(`/collections/${collId}/search/text`, 'POST', { query, k, method });
      renderSearchResults(infoBar, container, result);
    } else {
      infoBar.textContent = 'No collections available. Create one first.';
    }
  } catch(e) {
    infoBar.textContent = 'Error: ' + e.message;
  }
});

function renderSearchResults(infoBar, container, result) {
  if (!result.success || !result.results || result.results.length === 0) {
    infoBar.textContent = 'No results found.';
    return;
  }
  infoBar.textContent = `Found ${result.total_results} results in ${(result.search_time * 1000).toFixed(1)}ms`;
  result.results.forEach(r => {
    const div = document.createElement('div');
    div.className = 'result-item';
    const vectorId = r.vector_id || r.id || '--';
    const metaStr = r.metadata ? JSON.stringify(r.metadata).slice(0, 120) : '--';
    const dist = typeof r.distance === 'number' ? (r.distance * 10000).toFixed(2) : r.distance;
    div.innerHTML = `<div><div class="id">${vectorId}</div><div class="meta">${metaStr}</div></div><div class="distance">${dist}</div>`;
    container.appendChild(div);
  });
}

// ==================== Similarity Explorer ====================
document.getElementById('sim-btn').addEventListener('click', async () => {
  const query = document.getElementById('sim-query').value;
  const k = parseInt(document.getElementById('sim-k').value);
  if (!query) return;

  try {
    const colls = await api('/collections');
    if (colls.success && colls.collections && colls.collections.length > 0) {
      const collId = colls.collections[0].collection_id || colls.collections[0].id;
      const result = await api(`/collections/${collId}/search/text`, 'POST', { query, k, method: 'brute' });
      renderSimilarityChart(result);
      renderSimilarityResults(result);
    }
  } catch(e) {
    console.error('Similarity error:', e);
  }
});

let _simChart = null;
function renderSimilarityChart(result) {
  const ctx = document.getElementById('sim-chart').getContext('2d');
  if (_simChart) _simChart.destroy();

  if (!result.success || !result.results || result.results.length === 0) {
    _simChart = new Chart(ctx, { type: 'scatter', data: { datasets: [] } });
    return;
  }

  const items = result.results;
  const maxDist = Math.max(...items.map(r => r.distance));
  const minDist = Math.min(...items.map(r => r.distance));
  const range = maxDist - minDist || 1;

  // Circular projection: closer items cluster toward center
  const points = items.map((r, i) => {
    const angle = (2 * Math.PI * i) / items.length;
    const radius = ((r.distance - minDist) / range) * 0.8;
    return { x: Math.cos(angle) * radius, y: Math.sin(angle) * radius };
  });

  _simChart = new Chart(ctx, {
    type: 'scatter',
    data: {
      datasets: [{
        label: 'Vectors',
        data: points,
        backgroundColor: items.map(r => {
          const intensity = Math.round(255 * (1 - (r.distance - minDist) / range));
          return `rgb(79, ${intensity}, 247)`;
        }),
        pointRadius: 10,
        pointHoverRadius: 14
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => {
              const item = items[ctx.dataIndex];
              return `ID: ${item.vector_id || item.id}  Distance: ${(item.distance * 10000).toFixed(2)}`;
            }
          }
        }
      },
      scales: {
        x: { min: -1, max: 1, grid: { color: '#f0f0f0' }, ticks: { display: false } },
        y: { min: -1, max: 1, grid: { color: '#f0f0f0' }, ticks: { display: false } }
      }
    }
  });
}

function renderSimilarityResults(result) {
  const container = document.getElementById('sim-results');
  if (!result.success || !result.results || result.results.length === 0) {
    container.innerHTML = '<p>No results</p>';
    return;
  }
  container.innerHTML = '';
  result.results.forEach(r => {
    const div = document.createElement('div');
    div.className = 'result-item';
    const vectorId = r.vector_id || r.id || '--';
    div.innerHTML = `<div><div class="id">${vectorId}</div></div><div class="distance">${(r.distance * 10000).toFixed(2)}</div>`;
    container.appendChild(div);
  });
}

// ==================== Collections Tab ====================
document.getElementById('coll-create-btn').addEventListener('click', async () => {
  const name = document.getElementById('coll-name').value;
  const modality = document.getElementById('coll-modality').value;
  if (!name) return;
  try {
    const result = await api('/collections', 'POST', { name, modality });
    if (result.success) loadCollections();
  } catch(e) { console.error('Create collection error:', e); }
});

async function loadCollections() {
  try {
    const result = await api('/collections');
    const container = document.getElementById('coll-list');
    if (result.success && result.collections && result.collections.length > 0) {
      container.innerHTML = result.collections.map(c =>
        `<div class="coll-item">
          <div>
            <div class="name">${c.name}</div>
            <div class="meta">${c.collection_id || c.id} &middot; ${c.modality} &middot; ${c.vector_count || 0} vectors</div>
          </div>
        </div>`
      ).join('');
    } else {
      container.innerHTML = '<p>No collections yet. Create one above.</p>';
    }
  } catch(e) { console.error('Load collections error:', e); }
}

// ==================== Memories Tab ====================
document.getElementById('mem-search-btn').addEventListener('click', async () => {
  const query = document.getElementById('mem-search-query').value;
  if (!query) { loadMemories(); return; }
  try {
    const result = await api('/memories/search', 'POST', { query, limit: 50 });
    renderMemories(result.results || []);
    document.getElementById('mem-search-query').value = '';
  } catch(e) { console.error('Memory search error:', e); }
});

document.getElementById('mem-add-btn').addEventListener('click', async () => {
  const text = document.getElementById('mem-new-text').value;
  const cats = document.getElementById('mem-new-cats').value;
  if (!text) return;
  try {
    const categories = cats ? cats.split(',').map(c => c.trim()).filter(Boolean) : [];
    await api('/memories', 'POST', { text, categories });
    document.getElementById('mem-new-text').value = '';
    document.getElementById('mem-new-cats').value = '';
    loadMemories();
  } catch(e) { console.error('Memory add error:', e); }
});

async function loadMemories() {
  try {
    const stats = await api('/memories/stats');
    if (stats.success) {
      document.getElementById('mem-stat-total').textContent = stats.total.toLocaleString();
      document.getElementById('mem-stat-categories').textContent = Object.keys(stats.by_category).length;
      document.getElementById('mem-stat-7d').textContent = stats.recent_7d;
      document.getElementById('mem-stat-30d').textContent = stats.recent_30d;
    }
    const list = await api('/memories?limit=50');
    renderMemories(list.success ? list.memories : []);
  } catch(e) { console.error('Load memories error:', e); }
}

function renderMemories(memories) {
  const container = document.getElementById('mem-list');
  if (!memories.length) {
    container.innerHTML = '<p>No memories yet. Add one above.</p>';
    return;
  }
  container.innerHTML = memories.map(m => `
    <div class="coll-item">
      <div>
        <div class="name">${escapeHtml(m.text)}</div>
        <div class="meta">${m.memory_id} &middot; ${(m.categories || []).join(', ') || 'uncategorized'} &middot; ${m.created_at || ''}</div>
      </div>
    </div>
  `).join('');
}

function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

// ==================== AI Search Tab ====================
document.getElementById('nl-query-btn').addEventListener('click', async () => {
  const query = document.getElementById('nl-query-input').value;
  if (!query) return;
  const preview = document.getElementById('nl-query-preview');
  const container = document.getElementById('nl-query-results');
  preview.style.display = 'block';
  preview.textContent = 'Analyzing...';
  container.innerHTML = '';
  try {
    const result = await api('/search/nl', 'POST', { query, limit: 10 });
    if (result.success) {
      preview.textContent = 'Structured query: ' + JSON.stringify(result.structured_query, null, 2);
      if (result.results && result.results.length > 0) {
        container.innerHTML = result.results.map(r =>
          `<div class="result-item"><div><div class="id">${r.vector_id || r.id || r.memory_id || '--'}</div><div class="meta">${JSON.stringify(r.metadata || r).slice(0, 150)}</div></div><div class="distance">${r.distance ? (r.distance * 10000).toFixed(2) : '--'}</div></div>`
        ).join('');
      } else {
        container.innerHTML = '<p>No results found.</p>';
      }
    } else {
      preview.textContent = 'Error: ' + (result.message || 'Unknown error');
    }
  } catch(e) {
    preview.textContent = 'Error: ' + e.message;
  }
});

// ==================== Enterprise Tab ====================
document.getElementById('ent-set-retention').addEventListener('click', async () => {
  const collId = document.getElementById('ent-collection-id').value;
  const ttl = parseInt(document.getElementById('ent-ttl-days').value);
  if (!collId) return;
  try {
    const result = await api('/admin/retention', 'POST', { collection_id: collId, ttl_days: ttl });
    if (result.success) loadEnterprise();
  } catch(e) { console.error('Retention error:', e); }
});

document.getElementById('ent-gen-report').addEventListener('click', async () => {
  try {
    const result = await api('/admin/compliance/reports', 'POST', { report_type: 'SOC2', tenant_id: 'default' });
    if (result.success) loadEnterprise();
  } catch(e) { console.error('Report error:', e); }
});

async function loadEnterprise() {
  try {
    const policies = await api('/admin/retention/demo-collection');
    document.getElementById('ent-stat-policies').textContent = policies.policy ? '1 configured' : '0';
    const budgets = await api('/admin/query-budgets/default');
    document.getElementById('ent-stat-budgets').textContent = budgets.budget ? '1 configured' : '0';
    const reports = await api('/admin/compliance/reports/default');
    document.getElementById('ent-stat-reports').textContent = reports.reports ? reports.reports.length : 0;
    const container = document.getElementById('ent-results');
    if (reports.reports && reports.reports.length > 0) {
      container.innerHTML = reports.reports.map(r =>
        `<div class="coll-item"><div><div class="name">${r.report_type} Report</div><div class="meta">${r.status} &middot; ${r.generated_at || ''}</div></div></div>`
      ).join('');
    } else {
      container.innerHTML = '<p>No reports yet. Generate one above.</p>';
    }
  } catch(e) { console.error('Enterprise load error:', e); }
}

// ==================== Performance Tab ====================
document.getElementById('perf-run-benchmark').addEventListener('click', async () => {
  const container = document.getElementById('perf-results');
  container.innerHTML = '<p>Running benchmark...</p>';
  try {
    const result = await api('/admin/benchmark/run', 'POST');
    if (result.success) {
      container.innerHTML = result.results.map(r =>
        `<div class="coll-item"><div><div class="name">${r.method}</div><div class="meta">Recall: ${(r.avg_recall * 100).toFixed(1)}% &middot; Latency: ${r.avg_latency_ms}ms &middot; Queries: ${r.queries_run}</div></div></div>`
      ).join('');
    }
  } catch(e) { console.error('Benchmark error:', e); }
});

document.getElementById('perf-flush-cache').addEventListener('click', async () => {
  try {
    await api('/admin/cache', 'DELETE');
    loadPerformance();
  } catch(e) { console.error('Flush error:', e); }
});

document.getElementById('perf-refresh').addEventListener('click', loadPerformance);

async function loadPerformance() {
  try {
    const cache = await api('/admin/cache/stats');
    if (cache.success) {
      document.getElementById('perf-cache-hits').textContent = cache.stats.hits;
      document.getElementById('perf-cache-ratio').textContent = (cache.stats.hit_ratio * 100).toFixed(1) + '%';
    }
    const views = await api('/admin/views');
    document.getElementById('perf-views').textContent = views.views ? views.views.length : 0;
    const bench = await api('/admin/benchmark/results');
    document.getElementById('perf-benchmarks').textContent = bench.benchmarks ? bench.benchmarks.length : 0;
  } catch(e) { console.error('Perf load error:', e); }
}

// ==================== Monitoring Tab ====================
document.getElementById('mon-refresh').addEventListener('click', loadMonitoring);

async function loadMonitoring() {
  try {
    const stats = await api('/monitoring/slow-queries/stats');
    if (stats.success) {
      document.getElementById('mon-slow-total').textContent = stats.stats.total;
      document.getElementById('mon-avg-latency').textContent = stats.stats.avg_latency_ms + 'ms';
      document.getElementById('mon-p95-latency').textContent = stats.stats.p95_latency_ms + 'ms';
    }
    const health = await api('/monitoring/health/details');
    if (health.success) {
      document.getElementById('mon-cpu').textContent = health.health.cpu_percent + '%';
    }
    const slow = await api('/monitoring/slow-queries?limit=20');
    const container = document.getElementById('mon-results');
    if (slow.success && slow.slow_queries && slow.slow_queries.length > 0) {
      container.innerHTML = slow.slow_queries.map(q =>
        `<div class="coll-item"><div><div class="name">${q.collection_id} / ${q.method}</div><div class="meta">${q.latency_ms}ms &middot; ${q.timestamp}</div></div></div>`
      ).join('');
    } else {
      container.innerHTML = '<p>No slow queries recorded.</p>';
    }
  } catch(e) { console.error('Monitoring load error:', e); }
}

// ==================== Integrations Tab ====================
document.getElementById('int-enrich-btn').addEventListener('click', async () => {
  const text = document.getElementById('int-enrich-text').value;
  if (!text) return;
  const preview = document.getElementById('int-enrich-result');
  preview.style.display = 'block';
  preview.textContent = 'Enriching...';
  try {
    const result = await api('/enrich/metadata', 'POST', { text });
    if (result.success) {
      preview.textContent = JSON.stringify(result.enriched_metadata, null, 2);
    }
  } catch(e) { preview.textContent = 'Error: ' + e.message; }
});

async function loadIntegrations() {
  try {
    const models = await api('/admin/models');
    if (models.success) {
      document.getElementById('int-models').textContent = models.models.length;
      const active = models.models.find(m => m.status === 'active');
      document.getElementById('int-active-model').textContent = active ? active.model_id : 'None';
      const container = document.getElementById('int-models-list');
      container.innerHTML = models.models.map(m =>
        `<div class="coll-item"><div><div class="name">${m.model_id}</div><div class="meta">${m.provider} &middot; ${m.dimension}d &middot; ${m.status}</div></div></div>`
      ).join('');
    }
  } catch(e) { console.error('Integrations load error:', e); }
}

// ==================== Init ====================
document.addEventListener('DOMContentLoaded', () => {
  loadOverview();
  loadCollections();
});
