const datasetSelect = document.getElementById('dataset');
const queryInput = document.getElementById('query');
const modelSelect = document.getElementById('model');
const topKInput = document.getElementById('top_k');
const refineToggle = document.getElementById('refineToggle');
const preprocessMethod = document.getElementById('preprocessMethod');
const searchBtn = document.getElementById('searchBtn');
const clearQueryBtn = document.getElementById('clearQueryBtn');
const resultsGrid = document.getElementById('resultsGrid');
const navButtons = document.querySelectorAll('.nav-link');
const pageSections = document.querySelectorAll('.page-section');

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

const fetchSearch = async (query, model, top_k, refine, preprocessing) => {
    const response = await fetch('/api/search', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            query,
            model,
            top_k,
            refine,
            preprocessing,
            dataset: datasetSelect.value,
            weights: [0.33, 0.33, 0.34],
        }),
    });
    return response.json();
};

const fetchRefine = async (query) => {
    const response = await fetch('/api/refine', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
    });
    return response.json();
};

const fetchEvaluate = async (retrieved, relevant, top_k) => {
    const response = await fetch('/api/evaluate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ retrieved, relevant, top_k }),
    });
    return response.json();
};

const renderResults = (items) => {
    resultsGrid.innerHTML = '';
    if (!items || !items.length) {
        resultsGrid.innerHTML = '<div class="result-card"><h3>لا توجد نتائج</h3><p>جرب استعلاماً مختلفاً أو نموذجاً آخر.</p></div>';
        return;
    }

    items.forEach((item, index) => {
        const card = document.createElement('article');
        card.className = 'result-card';
        card.innerHTML = `
            <h3>النتيجة ${index + 1}</h3>
            <p><strong>معرف المستند:</strong> ${item.doc_id}</p>
            <p>${item.text ? item.text.slice(0, 180) + '...' : 'لا تتوفر معاينة نصية.'}</p>
            <div class="result-badge">النتيجة: ${item.score.toFixed(4)}</div>
        `;
        resultsGrid.appendChild(card);
    });
};

searchBtn.addEventListener('click', async () => {
    const query = queryInput.value.trim();
    const model = modelSelect.value;
    const top_k = Number(topKInput.value);
    const refine = refineToggle.checked;
    const preprocessing = preprocessMethod.value;

    if (!query) {
        alert('يرجى كتابة استعلام صالح.');
        return;
    }

    const response = await fetchSearch(query, model, top_k, refine, preprocessing);

    if (response.error) {
        resultsGrid.innerHTML = `<div class="result-card"><h3>خطأ</h3><p>${response.error}</p></div>`;
        return;
    }

    renderResults(response.results);
});

clearQueryBtn.addEventListener('click', () => {
    queryInput.value = '';
    queryInput.focus();
});
