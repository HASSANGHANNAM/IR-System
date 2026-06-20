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
const evaluationPanel = document.getElementById('evaluationPanel');
const searchBtn = document.getElementById('searchBtn');
const clearQueryBtn = document.getElementById('clearQueryBtn');
const resultsGrid = document.getElementById('resultsGrid');
const resultCount = document.getElementById('resultCount');
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

let lastSearchResponse = null;
let lastQueryId = null;

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

// ===== دوال الـ Hybrid =====

// إظهار/إخفاء قسم الـ Hybrid حسب النموذج المختار
function toggleHybridConfig() {
    if (modelSelect.value === 'hybrid') {
        hybridConfig.style.display = 'block';
    } else {
        hybridConfig.style.display = 'none';
    }
}

// تعطيل/تفعيل الأوزان والأزرار حسب وضع الـ Mode (Parallel / Serial)
function toggleWeightsSection() {
    const isSerial = hybridMode.value === 'serial';

    // تعطيل/تفعيل الأوزان
    weightsSection.style.opacity = isSerial ? '0.5' : '1';
    weightsSection.style.pointerEvents = isSerial ? 'none' : 'auto';

    // تعطيل/تفعيل أزرار الـ Presets
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

    // إظهار/إخفاء رسالة Serial
    serialNote.style.display = isSerial ? 'block' : 'none';
}

// تطبيع الأوزان (تشتت تلقائياً عند تغيير أي قيمة)
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

// ضبط الأوزان بالإعدادات السريعة
function setPreset(type) {
    if (type === 'equal') {
        wTfidf.value = 0.33;
        wBert.value = 0.33;
        wBm25.value = 0.34;
    } else if (type === 'best') {
        wTfidf.value = 0.30;
        wBert.value = 0.10;
        wBm25.value = 0.60;
    }
    normalizeWeights();
}

// ربط الأحداث الخاصة بالـ Hybrid
modelSelect.addEventListener('change', toggleHybridConfig);
hybridMode.addEventListener('change', toggleWeightsSection);

// ربط التطبيع عند تغيير أي من حقول الأوزان
[wTfidf, wBert, wBm25].forEach(input => {
    input.addEventListener('input', normalizeWeights);
});

// تشغيل الحالة الأولية
toggleHybridConfig();
toggleWeightsSection();

// ===== Fetch Search (معدل لدعم الـ Hybrid) =====
const fetchSearch = async (query, query_id, model, top_k, refine, preprocessing, evaluate, hybrid_mode, weights) => {
    try {
        const body = {
            query,
            query_id,
            model,
            top_k,
            refine,
            preprocessing,
            evaluate: evaluate || false,
            dataset: datasetSelect.value,
        };

        // إضافة بيانات الـ Hybrid إن وُجدت
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

// ===== Render Results (مع التقييم) =====
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
        return;
    }

    resultCount.textContent = items.length;

    items.forEach((item, index) => {
        const card = document.createElement('article');
        card.className = 'result-card';

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
    `;
};

// ===== Render Missing Relevant Docs =====
const renderMissingDocs = (missingItems) => {
    if (!missingDocsList) return;
    const missing = Array.isArray(missingItems) ? missingItems : [];

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
        return;
    }

    evaluationPanel.style.display = 'block';

    if (response.metrics) {
        renderMetrics(response.metrics);
    } else {
        metricsContainer.innerHTML = `
            <div class="metric-card"><strong>Precision@K</strong><span>0%</span></div>
            <div class="metric-card"><strong>MAP</strong><span>0%</span></div>
            <div class="metric-card"><strong>NDCG</strong><span>0%</span></div>
        `;
    }

    if (response.missing_relevant_docs !== undefined) {
        renderMissingDocs(response.missing_relevant_docs);
    } else {
        missingDocsList.innerHTML = '<p style="color: #6a7aac;">No evaluation data available for this query.</p>';
    }
};

// ===== Handle Search (معدل) =====
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

    // 🔥 التنبيه الفوري
    alert('🔍 Searching... Please wait.');

    const model = modelSelect.value;
    const top_k = Number(topKInput.value);
    const refine = refineToggle.checked;
    const preprocessing = preprocessMethod.value;

    // جمع بيانات الـ Hybrid إن كان النموذج Hybrid
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
            hybrid_mode, weights
        );

        if (response.error) {
            resultsGrid.innerHTML = `
                <div class="empty-state">
                    <span>⚠️</span>
                    <p>${response.error}</p>
                </div>
            `;
            resultCount.textContent = '0';
            evaluationPanel.style.display = 'none';
            lastSearchResponse = null;
            return;
        }

        lastSearchResponse = response;
        renderResults(response.results, evaluate);
        updateEvaluationPanel(response);

        if (evaluate && response.metrics) {
            renderMetrics(response.metrics);
            renderMissingDocs(response.missing_relevant_docs);
        }

    } catch (error) {
        resultsGrid.innerHTML = `
            <div class="empty-state">
                <span>❌</span>
                <p>Network error: ${error.message}</p>
            </div>
        `;
        resultCount.textContent = '0';
        evaluationPanel.style.display = 'none';
        lastSearchResponse = null;
    }
});

// ===== Clear Query =====
clearQueryBtn.addEventListener('click', () => {
    queryInput.value = '';
    queryInput.focus();
});

// ===== Evaluation Toggle =====
if (evalToggle) {
    evalToggle.addEventListener('change', () => {
        if (evalToggle.checked) {
            if (lastSearchResponse) {
                evaluationPanel.style.display = 'block';
                renderResults(lastSearchResponse.results, true);
                updateEvaluationPanel(lastSearchResponse);
            } else {
                evaluationPanel.style.display = 'block';
                metricsContainer.innerHTML = `
                    <div class="metric-card"><strong>Precision@K</strong><span>0%</span></div>
                    <div class="metric-card"><strong>MAP</strong><span>0%</span></div>
                    <div class="metric-card"><strong>NDCG</strong><span>0%</span></div>
                `;
                missingDocsList.innerHTML = '<p style="color: #6a7aac;">🔍 Perform a search to see evaluation results.</p>';
            }
        } else {
            evaluationPanel.style.display = 'none';
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

console.log('✅ IR-System ready with Hybrid support.');