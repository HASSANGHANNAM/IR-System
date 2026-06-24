console.log('✅ script.js loaded');

// ===== عناصر DOM =====
const datasetSelect = document.getElementById('dataset');
const queryInput = document.getElementById('query');
const modelSelect = document.getElementById('model');
const topKInput = document.getElementById('top_k');
const refineToggle = document.getElementById('refineToggle');
const preprocessMethod = document.getElementById('preprocessMethod');
const querySelect = document.getElementById('querySelect');
const evalToggle = document.getElementById('evalToggle');
const metricsContainer = document.getElementById('metricsContainer');
const missingDocsList = document.getElementById('missingDocsList');
const missingCount = document.getElementById('missingCount');
const evaluationPanel = document.getElementById('evaluationPanel');
const searchBtn = document.getElementById('searchBtn');
const clearQueryBtn = document.getElementById('clearQueryBtn');
const resultsGrid = document.getElementById('resultsGrid');
const resultCount = document.getElementById('resultCount');
const relevantCount = document.getElementById('relevantCount');
const nonRelevantCount = document.getElementById('nonRelevantCount');
const loadingSpinner = document.getElementById('loadingSpinner');
const navButtons = document.querySelectorAll('.nav-link');
const pageSections = document.querySelectorAll('.page-section');

// ===== عناصر الـ Hybrid =====
const hybridConfig = document.getElementById('hybridConfig');
const hybridMode = document.getElementById('hybridMode');
const wTfidf = document.getElementById('wTfidf');
const wBert = document.getElementById('wBert');
const wBm25 = document.getElementById('wBm25');
const weightsSection = document.getElementById('weightsSection');
const serialNote = document.getElementById('serialNote');
const presetEqual = document.getElementById('presetEqual');
const presetBest = document.getElementById('presetBest');

// ===== عناصر BM25 Configuration =====
const bm25Config = document.getElementById('bm25Config');
const bm25K1 = document.getElementById('bm25K1');
const bm25B = document.getElementById('bm25B');
const bm25PresetDefault = document.getElementById('bm25PresetDefault');
const bm25PresetBest = document.getElementById('bm25PresetBest');

// ===== عناصر FAISS =====
const faissToggle = document.getElementById('faissToggle');

// ===== عناصر Modal =====
const docModal = document.getElementById('docModal');
const modalTitle = document.getElementById('modalTitle');
const modalBody = document.getElementById('modalBody');
const closeModalBtn = document.getElementById('closeModalBtn');

// ===== عناصر Charts =====
const chartTypeSelect = document.getElementById('chartType');
const generateChartBtn = document.getElementById('generateChartBtn');
const chartContainer = document.getElementById('chartContainer');
const analysisText = document.getElementById('analysisText');

// 🔥 حاوية إحصاءات التقييم (للتحكم في إظهار/إخفاء الكل معاً)
const evalStats = document.getElementById('evalStats');

let lastSearchResponse = null;
let lastQueryId = null;
let chartInstance = null;

// ===== Navigation =====
const switchPage = (targetId) => {
    pageSections.forEach((section) => {
        section.classList.toggle('active', section.id === targetId);
    });
    navButtons.forEach((button) => {
        button.classList.toggle('active', button.dataset.target === targetId);
    });
};

navButtons.forEach((button) => {
    button.addEventListener('click', () => switchPage(button.dataset.target));
});

// ===== Load Queries =====
const loadQueries = async () => {
    if (!querySelect) return;
    try {
        const response = await fetch('/api/queries');
        if (!response.ok) throw new Error(`Failed to load queries (${response.status})`);
        const payload = await response.json();
        const queries = Array.isArray(payload.queries) ? payload.queries : [];

        querySelect.innerHTML = '<option value="">-- Select a saved query --</option>';
        queries.forEach((item) => {
            const option = document.createElement('option');
            option.value = item.query_id || '';
            option.dataset.queryText = item.text || '';
            option.textContent = `${item.query_id}: ${item.text || ''}`;
            querySelect.appendChild(option);
        });
    } catch (error) {
        console.error('Query loading error:', error);
        querySelect.innerHTML = '<option value="">Unable to load queries</option>';
    }
};

if (querySelect) {
    querySelect.addEventListener('change', () => {
        const selectedOption = querySelect.selectedOptions[0];
        if (selectedOption && selectedOption.dataset.queryText) {
            queryInput.value = selectedOption.dataset.queryText;
            lastQueryId = selectedOption.value;
        }
        querySelect.classList.remove('show');
        querySelect.style.opacity = '0';
        querySelect.style.pointerEvents = 'none';
    });
}

loadQueries();

// ===== Dropdown Toggle (▼) =====
const dropdownToggleBtn = document.getElementById('dropdownToggleBtn');
if (dropdownToggleBtn && querySelect) {
    dropdownToggleBtn.addEventListener('click', function (e) {
        e.stopPropagation();

        if (querySelect.classList.contains('show')) {
            querySelect.classList.remove('show');
            querySelect.style.opacity = '0';
            querySelect.style.pointerEvents = 'none';
            return;
        }

        if (typeof querySelect.showPicker === 'function') {
            try {
                querySelect.showPicker();
                return;
            } catch (err) {
                console.warn('showPicker failed, using fallback:', err);
            }
        }

        querySelect.classList.add('show');
        querySelect.style.opacity = '1';
        querySelect.style.pointerEvents = 'auto';
        querySelect.focus();
    });

    querySelect.addEventListener('blur', function () {
        setTimeout(() => {
            querySelect.classList.remove('show');
            querySelect.style.opacity = '0';
            querySelect.style.pointerEvents = 'none';
        }, 150);
    });

    querySelect.addEventListener('keydown', function (e) {
        if (e.key === 'Escape') {
            querySelect.classList.remove('show');
            querySelect.style.opacity = '0';
            querySelect.style.pointerEvents = 'none';
            querySelect.blur();
        }
    });
}

// ============================================================
// ===== دوال الـ Hybrid =====
// ============================================================

function toggleHybridConfig() {
    if (modelSelect.value === 'hybrid') {
        hybridConfig.style.display = 'block';
    } else {
        hybridConfig.style.display = 'none';
    }
}

function toggleWeightsSection() {
    const isSerial = hybridMode.value === 'serial';
    weightsSection.style.opacity = isSerial ? '0.5' : '1';
    weightsSection.style.pointerEvents = isSerial ? 'none' : 'auto';
    const presets = [presetEqual, presetBest];
    presets.forEach(btn => {
        if (isSerial) {
            btn.disabled = true;
            btn.style.opacity = '0.5';
            btn.style.cursor = 'not-allowed';
        } else {
            btn.disabled = false;
            btn.style.opacity = '1';
            btn.style.cursor = 'pointer';
        }
    });
    serialNote.style.display = isSerial ? 'block' : 'none';
}

function normalizeWeights() {
    let a = parseFloat(wTfidf.value) || 0;
    let b = parseFloat(wBert.value) || 0;
    let c = parseFloat(wBm25.value) || 0;
    let sum = a + b + c;
    if (sum > 0) {
        wTfidf.value = Math.round((a / sum) * 100) / 100;
        wBert.value = Math.round((b / sum) * 100) / 100;
        wBm25.value = Math.round((c / sum) * 100) / 100;
    }
}

function setPreset(type) {
    if (type === 'equal') {
        wTfidf.value = 0.33;
        wBert.value = 0.33;
        wBm25.value = 0.34;
    } else if (type === 'best') {
        wTfidf.value = 0.20;
        wBert.value = 0.00;
        wBm25.value = 0.80;
    }
    normalizeWeights();
}

// ============================================================
// ===== دوال BM25 Configuration =====
// ============================================================

function toggleBm25Config() {
    if (modelSelect.value === 'bm25') {
        bm25Config.style.display = 'block';
    } else {
        bm25Config.style.display = 'none';
    }
}

function setBm25Preset(type) {
    if (type === 'default') {
        bm25K1.value = '1.5';
        bm25B.value = '0.75';
    } else if (type === 'best') {
        bm25K1.value = '2.0';
        bm25B.value = '0.60';
    }
}

// ============================================================
// ===== ربط الأحداث =====
// ============================================================

modelSelect.addEventListener('change', function () {
    toggleHybridConfig();
    toggleBm25Config();
});

hybridMode.addEventListener('change', toggleWeightsSection);

[wTfidf, wBert, wBm25].forEach(input => {
    input.addEventListener('input', normalizeWeights);
});

toggleHybridConfig();
toggleBm25Config();
toggleWeightsSection();

// ============================================================
// ===== Fetch Search =====
// ============================================================

const fetchSearch = async (query, query_id, model, top_k, refine, preprocessing, evaluate, hybrid_mode, weights, useFaiss) => {
    try {
        const bm25_k1 = parseFloat(bm25K1.value) || 1.5;
        const bm25_b = parseFloat(bm25B.value) || 0.75;

        const body = {
            query,
            query_id,
            model,
            top_k,
            refine,
            preprocessing,
            evaluate: evaluate || false,
            dataset: datasetSelect.value,
            bm25_k1: bm25_k1,
            bm25_b: bm25_b,
            use_faiss: useFaiss || false
        };

        if (model === 'hybrid') {
            body.hybrid_mode = hybrid_mode || 'parallel';
            if (hybrid_mode === 'parallel' && weights) {
                body.weights = weights;
            }
        }

        const response = await fetch('/api/search', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`Request failed (${response.status}): ${errorText}`);
        }
        return response.json();
    } catch (error) {
        console.error('Search request error:', error);
        throw error;
    }
};

// ============================================================
// ===== دوال عرض النتائج (معدلة) =====
// ============================================================

// 🔥 دالة مساعدة لإظهار/إخفاء إحصاءات التقييم
function toggleEvalStats(show) {
    if (evalStats) {
        evalStats.style.display = show ? 'flex' : 'none';
    } else {
        // حل احتياطي في حال عدم وجود الحاوية
        if (relevantCount) relevantCount.style.display = show ? 'inline-block' : 'none';
        if (nonRelevantCount) nonRelevantCount.style.display = show ? 'inline-block' : 'none';
    }
}

const renderResults = (items, isEval) => {
    resultsGrid.innerHTML = '';
    if (!items || !items.length) {
        resultsGrid.innerHTML = `
            <div class="empty-state">
                <span>🔍</span>
                <p>No results found. Try a different query.</p>
            </div>
        `;
        resultCount.textContent = '0';
        relevantCount.textContent = '✅ 0';
        nonRelevantCount.textContent = '❌ 0';
        toggleEvalStats(false); // 🔥 إخفاء الإحصاءات
        return;
    }

    resultCount.textContent = items.length;

    // 🔥 حساب الـ relevant و non‑relevant فقط إذا كان التقييم مفعلاً
    let relevant = 0;
    let nonRelevant = 0;
    if (isEval) {
        items.forEach(item => {
            if (item.relevant === true) relevant++;
            else if (item.relevant === false) nonRelevant++;
        });
    }
    relevantCount.textContent = `✅ ${relevant}`;
    nonRelevantCount.textContent = `❌ ${nonRelevant}`;

    // 🔥 إظهار أو إخفاء الإحصاءات حسب حالة التقييم
    toggleEvalStats(isEval);

    items.forEach((item, index) => {
        const card = document.createElement('article');
        card.className = 'result-card';
        card.dataset.docId = item.doc_id;

        let badgeHtml = '';
        let relevanceHtml = '';
        if (isEval && item.relevant !== undefined) {
            badgeHtml = `
                <span class="result-badge" style="color:${item.relevant ? '#4caf50' : '#f44336'};">
                    ${item.relevant ? '✅' : '❌'} ${item.relevant ? 'Relevant' : 'Not Relevant'}
                </span>
            `;
            relevanceHtml = `
                <div class="result-relevance">
                    Relevance score: ${item.relevance_score !== undefined ? item.relevance_score : 'N/A'}
                </div>
            `;
        }

        card.innerHTML = `
            <div class="result-header">
                <span class="result-title">📄 Result ${index + 1}</span>
                <span class="result-score">${item.score.toFixed(4)}</span>
                ${badgeHtml}
            </div>
            <div class="result-text">${item.text ? item.text.slice(0, 300) + '...' : 'No text available'}</div>
            <div class="result-id">Doc ID: ${item.doc_id}</div>
            ${relevanceHtml}
        `;
        resultsGrid.appendChild(card);
    });
};

// ===== Render Evaluation Metrics =====
const renderMetrics = (metrics) => {
    if (!metricsContainer) return;
    metricsContainer.innerHTML = `
        <div class="metric-card"><strong>Precision@K</strong><span>${(metrics.precision_at_k * 100).toFixed(2)}%</span></div>
        <div class="metric-card"><strong>MAP</strong><span>${(metrics.map * 100).toFixed(2)}%</span></div>
        <div class="metric-card"><strong>NDCG</strong><span>${(metrics.ndcg * 100).toFixed(2)}%</span></div>
        <div class="metric-card"><strong>Recall@K</strong><span>${(metrics.recall_at_k * 100).toFixed(2)}%</span></div>
    `;
};

// ===== Render Missing Relevant Docs =====
const renderMissingDocs = (missingItems) => {
    if (!missingDocsList) return;
    const missing = Array.isArray(missingItems) ? missingItems : [];

    if (missingCount) {
        missingCount.textContent = missing.length;
    }

    if (!missing.length) {
        missingDocsList.innerHTML = '<p style="color: #6a7aac;">✅ All relevant docs were retrieved within the top-K.</p>';
        return;
    }

    let html = '';
    missing.forEach((doc) => {
        html += `
            <div class="missing-doc-card">
                <div class="doc-id">📄 ${doc.doc_id}</div>
                ${doc.title ? `<div class="doc-title">${doc.title}</div>` : ''}
            </div>
        `;
    });
    missingDocsList.innerHTML = html;
};

// ===== Update Evaluation Panel =====
const updateEvaluationPanel = (response) => {
    const isEval = evalToggle && evalToggle.checked;

    if (!isEval || !response) {
        evaluationPanel.style.display = 'none';
        toggleEvalStats(false); // 🔥 إخفاء الإحصاءات
        return;
    }

    evaluationPanel.style.display = 'block';
    toggleEvalStats(true); // 🔥 إظهار الإحصاءات

    if (response.metrics) {
        renderMetrics(response.metrics);
    } else {
        metricsContainer.innerHTML = `
            <div class="metric-card"><strong>Precision@K</strong><span>0%</span></div>
            <div class="metric-card"><strong>MAP</strong><span>0%</span></div>
            <div class="metric-card"><strong>NDCG</strong><span>0%</span></div>
            <div class="metric-card"><strong>Recall@K</strong><span>0%</span></div>
        `;
    }

    if (response.missing_relevant_docs !== undefined) {
        renderMissingDocs(response.missing_relevant_docs);
    } else {
        missingDocsList.innerHTML = '<p style="color: #6a7aac;">No evaluation data available for this query.</p>';
        if (missingCount) missingCount.textContent = '0';
    }
};

// ============================================================
// ===== دوال Modal =====
// ============================================================

function openModal(docId) {
    docModal.style.display = 'flex';
    modalBody.innerHTML = 'جاري تحميل النص...';
    modalTitle.textContent = `📄 نص الوثيقة (${docId})`;

    fetch(`/api/document/${docId}`)
        .then(response => {
            if (!response.ok) {
                throw new Error('Failed to fetch document');
            }
            return response.json();
        })
        .then(data => {
            if (data.error) {
                modalBody.innerHTML = `<p style="color: #f44336;">${data.error}</p>`;
                return;
            }
            modalTitle.textContent = data.title ? `📄 ${data.title}` : `📄 نص الوثيقة (${data.doc_id || docId})`;
            modalBody.innerHTML = data.text || 'نص الوثيقة غير متوفر.';
        })
        .catch(error => {
            console.error('Error fetching document:', error);
            modalBody.innerHTML = '<p style="color: #f44336;">حدث خطأ أثناء تحميل النص.</p>';
        });
}

function closeModal() {
    docModal.style.display = 'none';
}

docModal.addEventListener('click', function (e) {
    if (e.target === this) {
        closeModal();
    }
});

closeModalBtn.addEventListener('click', closeModal);

document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape' && docModal.style.display === 'flex') {
        closeModal();
    }
});

resultsGrid.addEventListener('click', function (e) {
    const card = e.target.closest('.result-card');
    if (!card) return;
    const docId = card.dataset.docId;
    if (docId) {
        openModal(docId);
    }
});

// ============================================================
// ===== Handle Search (معدل) =====
// ============================================================

searchBtn.addEventListener('click', async () => {
    const query = queryInput.value.trim();
    if (!query) {
        alert('⚠️ Please enter a valid query.');
        return;
    }

    const selectedOption = querySelect ? querySelect.selectedOptions[0] : null;
    const query_id = selectedOption && selectedOption.value ? selectedOption.value : '';
    const evaluate = evalToggle ? evalToggle.checked : false;

    if (evaluate && !query_id) {
        alert('⚠️ Please select a saved query from the dropdown to enable evaluation (qrels).');
        return;
    }

    if (loadingSpinner) {
        loadingSpinner.style.display = 'block';
    }
    resultsGrid.innerHTML = '';

    const model = modelSelect.value;
    const top_k = Number(topKInput.value);
    const refine = refineToggle.checked;
    const preprocessing = preprocessMethod.value;
    const useFaiss = faissToggle ? faissToggle.checked : false;

    let hybrid_mode = null;
    let weights = null;
    if (model === 'hybrid') {
        hybrid_mode = hybridMode.value;
        if (hybrid_mode === 'parallel') {
            weights = {
                tfidf: parseFloat(wTfidf.value) || 0,
                bert: parseFloat(wBert.value) || 0,
                bm25: parseFloat(wBm25.value) || 0
            };
        }
    }

    try {
        const response = await fetchSearch(
            query, query_id, model, top_k, refine, preprocessing, evaluate,
            hybrid_mode, weights, useFaiss
        );

        if (loadingSpinner) {
            loadingSpinner.style.display = 'none';
        }

        if (response.error) {
            resultsGrid.innerHTML = `
                <div class="empty-state">
                    <span>⚠️</span>
                    <p>${response.error}</p>
                </div>
            `;
            resultCount.textContent = '0';
            relevantCount.textContent = '✅ 0';
            nonRelevantCount.textContent = '❌ 0';
            if (missingCount) missingCount.textContent = '0';
            evaluationPanel.style.display = 'none';
            toggleEvalStats(false); // 🔥 إخفاء الإحصاءات
            lastSearchResponse = null;
            return;
        }

        lastSearchResponse = response;
        renderResults(response.results, evaluate);
        updateEvaluationPanel(response);

        // 🔥 التأكد من إخفاء الإحصاءات إذا كان التقييم غير مفعّل
        toggleEvalStats(evaluate);

        if (evaluate && response.metrics) {
            renderMetrics(response.metrics);
            renderMissingDocs(response.missing_relevant_docs);
        }

    } catch (error) {
        if (loadingSpinner) {
            loadingSpinner.style.display = 'none';
        }
        resultsGrid.innerHTML = `
            <div class="empty-state">
                <span>❌</span>
                <p>Network error: ${error.message}</p>
            </div>
        `;
        resultCount.textContent = '0';
        relevantCount.textContent = '✅ 0';
        nonRelevantCount.textContent = '❌ 0';
        if (missingCount) missingCount.textContent = '0';
        evaluationPanel.style.display = 'none';
        toggleEvalStats(false);
        lastSearchResponse = null;
    }
});

// ===== Clear Query =====
clearQueryBtn.addEventListener('click', () => {
    queryInput.value = '';
    queryInput.focus();
});

// ============================================================
// ===== Evaluation Toggle (معدل) =====
// ============================================================

if (evalToggle) {
    evalToggle.addEventListener('change', () => {
        if (evalToggle.checked) {
            if (lastSearchResponse) {
                evaluationPanel.style.display = 'block';
                renderResults(lastSearchResponse.results, true);
                updateEvaluationPanel(lastSearchResponse);
                toggleEvalStats(true); // 🔥 إظهار الإحصاءات
            } else {
                evaluationPanel.style.display = 'block';
                metricsContainer.innerHTML = `
                    <div class="metric-card"><strong>Precision@K</strong><span>0%</span></div>
                    <div class="metric-card"><strong>MAP</strong><span>0%</span></div>
                    <div class="metric-card"><strong>NDCG</strong><span>0%</span></div>
                    <div class="metric-card"><strong>Recall@K</strong><span>0%</span></div>
                `;
                missingDocsList.innerHTML = '<p style="color: #6a7aac;">🔍 Perform a search to see evaluation results.</p>';
                if (missingCount) missingCount.textContent = '0';
                toggleEvalStats(true); // 🔥 إظهار الإحصاءات (حتى لو كانت 0)
            }
        } else {
            evaluationPanel.style.display = 'none';
            toggleEvalStats(false); // 🔥 إخفاء الإحصاءات
            if (lastSearchResponse) {
                renderResults(lastSearchResponse.results, false);
            }
        }
    });
}

// ===== Keyboard shortcut =====
queryInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
        searchBtn.click();
    }
});

// ===== Test evaluation from Console =====
window.testEvaluation = async (query_id = '1') => {
    const queryText = queryInput.value.trim() || 'Should teachers get tenure?';
    const response = await fetchSearch(queryText, query_id, 'tfidf', 10, false, 'stemming', true);
    console.log('🔍 Evaluation test response:', response);
    return response;
};

// ============================================================
// ===== دوال Charts (المخططات) =====
// ============================================================

async function fetchChartData(chartType) {
    const response = await fetch(`/api/chart/${chartType}`);
    if (!response.ok) {
        throw new Error(`Failed to fetch chart data: ${response.status}`);
    }
    return await response.json();
}

function renderAnalysis(chartType, data) {
    let analysis = '';
    switch (chartType) {
        case 'model_comparison':
            const bestModel = data.models.reduce((a, b) => data.map[data.models.indexOf(a)] > data.map[data.models.indexOf(b)] ? a : b);
            analysis = `
                <strong>🔍 Key Insights:</strong><br>
                • Best model by MAP: <strong>${bestModel}</strong> (${(data.map[data.models.indexOf(bestModel)] * 100).toFixed(2)}%)<br>
                • BM25 outperforms TF-IDF and BERT in all metrics, with MAP = ${(data.map[1] * 100).toFixed(2)}%.<br>
                • BERT has the lowest precision (${(data.precision[2] * 100).toFixed(2)}%) but fastest average time (${data.time[2].toFixed(2)}s).<br>
                • Consider Hybrid model for better balance of speed and accuracy.
            `;
            break;
        case 'bm25_params':
            let best = data.reduce((a, b) => a.avg_map > b.avg_map ? a : b);
            analysis = `
                <strong>🔍 Optimal BM25 Parameters:</strong><br>
                • Best k1 = ${best.k1}, b = ${best.b} with MAP = ${(best.avg_map * 100).toFixed(2)}%<br>
                • Higher b (length normalization) above 0.8 degrades performance significantly.<br>
                • k1 between 1.5 and 2.5 generally performs best across different b values.<br>
                • Total configurations tested: ${data.length}
            `;
            break;
        case 'hybrid_weights':
            let bestWeight = data.reduce((a, b) => a.map > b.map ? a : b);
            analysis = `
                <strong>🔍 Hybrid Weights Impact:</strong><br>
                • Best combination: TF-IDF=${bestWeight.tfidf}, BERT=${bestWeight.bert}, BM25=${bestWeight.bm25}<br>
                • MAP = ${(bestWeight.map * 100).toFixed(2)}%, NDCG = ${(bestWeight.ndcg * 100).toFixed(2)}%<br>
                • BM25 weight > 0.6 generally improves performance significantly.<br>
                • BERT alone (weight=1.0) performs poorly (MAP = ${(data.find(d => d.bert === 1.0 && d.tfidf === 0 && d.bm25 === 0)?.map || 0) * 100}%).
            `;
            break;
        case 'serial_permutations':
            let bestSeq = data.reduce((a, b) => a.map > b.map ? a : b);
            analysis = `
                <strong>🔍 Serial Permutations Analysis:</strong><br>
                • Best sequence: <strong>${bestSeq.sequence_name}</strong> with MAP = ${(bestSeq.map * 100).toFixed(2)}%<br>
                • BM25 alone performs better than most 2-model sequences.<br>
                • Adding BERT after TF-IDF often degrades performance.<br>
                • Total sequences tested: ${data.length}
            `;
            break;
        default:
            analysis = 'Analysis not available for this chart type.';
    }
    analysisText.innerHTML = analysis;
}

function renderChart(chartType, data) {
    if (chartInstance) {
        chartInstance.destroy();
        chartInstance = null;
    }

    chartContainer.innerHTML = '<canvas id="myChart"></canvas>';
    const ctx = document.getElementById('myChart').getContext('2d');

    let config = {};

    switch (chartType) {
        case 'model_comparison':
            config = {
                type: 'bar',
                data: {
                    labels: data.models.map(m => m.toUpperCase()),
                    datasets: [
                        {
                            label: 'Precision@K',
                            data: data.precision,
                            backgroundColor: 'rgba(78, 123, 255, 0.7)',
                            borderColor: 'rgba(78, 123, 255, 1)',
                            borderWidth: 2
                        },
                        {
                            label: 'MAP',
                            data: data.map,
                            backgroundColor: 'rgba(51, 217, 255, 0.7)',
                            borderColor: 'rgba(51, 217, 255, 1)',
                            borderWidth: 2
                        },
                        {
                            label: 'NDCG',
                            data: data.ndcg,
                            backgroundColor: 'rgba(198, 89, 255, 0.7)',
                            borderColor: 'rgba(198, 89, 255, 1)',
                            borderWidth: 2
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: true,
                    plugins: {
                        title: { display: true, text: 'Model Performance Comparison (Top-K=10)', color: '#c5d2ff', font: { size: 16 } },
                        legend: { labels: { color: '#c8d8ff' } }
                    },
                    scales: {
                        y: { beginAtZero: true, ticks: { color: '#8899bb', callback: (val) => (val * 100).toFixed(0) + '%' } },
                        x: { ticks: { color: '#c8d8ff' } }
                    }
                }
            };
            break;

        case 'bm25_params':
            const k1Groups = {};
            data.forEach(d => {
                const key = d.k1;
                if (!k1Groups[key]) k1Groups[key] = [];
                k1Groups[key].push({ b: d.b, map: d.avg_map });
            });
            const k1Keys = Object.keys(k1Groups).sort((a, b) => parseFloat(a) - parseFloat(b));
            const bValues = [...new Set(data.map(d => d.b))].sort();

            const datasets = k1Keys.map((k1, idx) => {
                const points = k1Groups[k1].sort((a, b) => a.b - b.b);
                const hue = (idx * 360) / k1Keys.length;
                return {
                    label: `k1 = ${k1}`,
                    data: points.map(p => p.map),
                    borderColor: `hsl(${hue}, 70%, 60%)`,
                    backgroundColor: `hsla(${hue}, 70%, 60%, 0.1)`,
                    fill: false,
                    tension: 0.1,
                    pointRadius: 3,
                    pointHoverRadius: 6
                };
            });

            config = {
                type: 'line',
                data: { labels: bValues.map(b => b.toFixed(2)), datasets: datasets },
                options: {
                    responsive: true,
                    maintainAspectRatio: true,
                    plugins: {
                        title: { display: true, text: 'BM25 Parameters Impact on MAP', color: '#c5d2ff', font: { size: 16 } },
                        legend: { labels: { color: '#c8d8ff' } }
                    },
                    scales: {
                        y: { beginAtZero: true, ticks: { color: '#8899bb', callback: (val) => (val * 100).toFixed(1) + '%' } },
                        x: { title: { display: true, text: 'b parameter', color: '#c8d8ff' }, ticks: { color: '#c8d8ff' } }
                    }
                }
            };
            break;

        case 'hybrid_weights':
            const sorted = [...data].sort((a, b) => b.map - a.map).slice(0, 15);
            const labels = sorted.map(d => `TF:${d.tfidf.toFixed(1)}, B:${d.bert.toFixed(1)}, BM:${d.bm25.toFixed(1)}`);
            const mapData = sorted.map(d => d.map);
            const ndcgData = sorted.map(d => d.ndcg);

            config = {
                type: 'bar',
                data: {
                    labels: labels,
                    datasets: [
                        { label: 'MAP', data: mapData, backgroundColor: 'rgba(78, 123, 255, 0.7)', borderColor: 'rgba(78, 123, 255, 1)', borderWidth: 1 },
                        { label: 'NDCG', data: ndcgData, backgroundColor: 'rgba(198, 89, 255, 0.7)', borderColor: 'rgba(198, 89, 255, 1)', borderWidth: 1 }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: true,
                    plugins: {
                        title: { display: true, text: 'Top 15 Hybrid Weight Combinations', color: '#c5d2ff', font: { size: 16 } },
                        legend: { labels: { color: '#c8d8ff' } }
                    },
                    scales: {
                        y: { beginAtZero: true, ticks: { color: '#8899bb', callback: (val) => (val * 100).toFixed(1) + '%' } },
                        x: { ticks: { color: '#c8d8ff', maxRotation: 45, font: { size: 8 } } }
                    }
                }
            };
            break;

        case 'serial_permutations':
            const sortedSeq = [...data].sort((a, b) => b.map - a.map).slice(0, 15);
            const seqLabels = sortedSeq.map(d => d.sequence_name);
            const seqMap = sortedSeq.map(d => d.map);
            const seqNdcg = sortedSeq.map(d => d.ndcg);

            config = {
                type: 'bar',
                data: {
                    labels: seqLabels,
                    datasets: [
                        { label: 'MAP', data: seqMap, backgroundColor: 'rgba(51, 217, 255, 0.7)', borderColor: 'rgba(51, 217, 255, 1)', borderWidth: 1 },
                        { label: 'NDCG', data: seqNdcg, backgroundColor: 'rgba(78, 123, 255, 0.7)', borderColor: 'rgba(78, 123, 255, 1)', borderWidth: 1 }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: true,
                    plugins: {
                        title: { display: true, text: 'Top 15 Serial Permutations Performance', color: '#c5d2ff', font: { size: 16 } },
                        legend: { labels: { color: '#c8d8ff' } }
                    },
                    scales: {
                        y: { beginAtZero: true, ticks: { color: '#8899bb', callback: (val) => (val * 100).toFixed(1) + '%' } },
                        x: { ticks: { color: '#c8d8ff', maxRotation: 30, font: { size: 9 } } }
                    }
                }
            };
            break;

        default:
            chartContainer.innerHTML = '<p style="color: #f44336; text-align: center;">⚠️ Unsupported chart type.</p>';
            return;
    }

    chartInstance = new Chart(ctx, config);
}

// ===== حدث Generate Chart =====
generateChartBtn.addEventListener('click', async () => {
    const chartType = chartTypeSelect.value;
    if (!chartType) {
        chartContainer.innerHTML = '<p style="color: #f44336;">Please select a chart type.</p>';
        return;
    }

    chartContainer.innerHTML = '<p style="color: #8899bb; text-align: center;">⏳ Loading chart data...</p>';
    analysisText.textContent = 'Loading analysis...';

    try {
        const data = await fetchChartData(chartType);
        renderChart(chartType, data);
        renderAnalysis(chartType, data);
    } catch (error) {
        console.error('Chart error:', error);
        chartContainer.innerHTML = `<p style="color: #f44336; text-align: center;">❌ Error: ${error.message}</p>`;
        analysisText.textContent = 'Failed to load chart data. Please try again.';
    }
});

console.log('✅ IR-System ready with Hybrid support and document modal.');