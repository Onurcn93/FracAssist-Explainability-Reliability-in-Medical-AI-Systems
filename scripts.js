/**
 * scripts.js — FracAssist UI logic
 *
 * Connects the Flask inference backend (inference/app.py) to the HTML.
 * Entry point: python inference/app.py → http://127.0.0.1:5000
 */

const API_URL = 'http://127.0.0.1:5000';

// ─── State ────────────────────────────────────────────────────────────────
let _resultData      = null;
let _currentOverlay  = 'box';
let _currentFilename = null;

// ─── Tab switching ─────────────────────────────────────────────────────────
document.querySelectorAll('.tab').forEach(tab => {
    tab.addEventListener('click', () => {
        document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        document.querySelectorAll('.tab-panel').forEach(p => p.classList.add('hidden'));
        document.getElementById('tab-' + tab.dataset.tab).classList.remove('hidden');
        if (tab.dataset.tab === 'review') _loadReviewQueue();
    });
});

// ─── Expert Review — diagnose panel ────────────────────────────────────────
function openDiagnose(btn, imageId) {
    document.getElementById('diagnose-panel-id').textContent = imageId;
    const img = document.getElementById('diagnose-img');
    img.src = `${API_URL}/fractatlas/${imageId}`;
    img.classList.remove('hidden');
    document.getElementById('diagnose-placeholder').classList.add('hidden');
    document.querySelectorAll('.review-row').forEach(r => r.classList.remove('review-row-active'));
    btn.closest('.review-row').classList.add('review-row-active');
}

function closeDiagnose() {
    document.getElementById('diagnose-panel-id').textContent = '—';
    const img = document.getElementById('diagnose-img');
    img.classList.add('hidden');
    img.src = '';
    document.getElementById('diagnose-placeholder').classList.remove('hidden');
    document.querySelectorAll('.review-row').forEach(r => r.classList.remove('review-row-active'));
}

// ─── File picker ───────────────────────────────────────────────────────────
document.getElementById('select-btn').addEventListener('click', () => {
    document.getElementById('file-input').click();
});

document.getElementById('file-input').addEventListener('change', e => {
    const file = e.target.files[0];
    if (file) handleFile(file);
    e.target.value = ''; // allow re-selecting the same file
});

// ─── Drag and drop ─────────────────────────────────────────────────────────
const imageDisplay = document.getElementById('image-display');

imageDisplay.addEventListener('dragover', e => {
    e.preventDefault();
    imageDisplay.classList.add('drag-over');
});

imageDisplay.addEventListener('dragleave', () => {
    imageDisplay.classList.remove('drag-over');
});

imageDisplay.addEventListener('drop', e => {
    e.preventDefault();
    imageDisplay.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) handleFile(file);
});

// ─── Overlay toggle (GradCAM / Bounding Box) ───────────────────────────────
document.querySelectorAll('input[name="view-toggle"]').forEach(radio => {
    radio.addEventListener('change', () => {
        _currentOverlay = radio.value;
        if (_resultData) _updateOverlay(_currentOverlay);
    });
});

// ─── Refresh / reset ───────────────────────────────────────────────────────
document.getElementById('refresh-btn').addEventListener('click', resetUI);

// ─── Zoom controls ─────────────────────────────────────────────────────────
document.getElementById('zoom-slider').addEventListener('input', e => {
    document.getElementById('result-img').style.transform = `scale(${e.target.value / 100})`;
    document.getElementById('zoom-label').textContent = e.target.value + '%';
});

document.getElementById('zoom-reset').addEventListener('click', _resetZoom);

function _resetZoom() {
    document.getElementById('zoom-slider').value = 100;
    document.getElementById('result-img').style.transform = 'scale(1)';
    document.getElementById('zoom-label').textContent = '100%';
}

// ─── File handling → POST /predict ─────────────────────────────────────────
function handleFile(file) {
    const allowed = ['image/jpeg', 'image/jpg', 'image/png'];
    if (!allowed.includes(file.type)) {
        showError('Invalid file type. Please use JPG or PNG.');
        return;
    }

    _resultData = null;
    _currentFilename = file.name;
    showState('loading');
    resetMetrics();
    _resetZoom();
    document.getElementById('img-id-badge').textContent = 'IMG · ' + file.name;

    const formData = new FormData();
    formData.append('image', file);
    formData.append('inference_mode', document.getElementById('model-select').value);

    fetch(`${API_URL}/predict`, { method: 'POST', body: formData })
        .then(res => res.json())
        .then(data => {
            if (data.error) { showError(data.error); return; }
            _resultData = data;
            applyPrediction(data);
        })
        .catch(() => {
            showError('Server unreachable — run: python inference/app.py');
        });
}

// ─── Apply prediction response to UI ──────────────────────────────────────
function applyPrediction(data) {
    const isFrac = data.label === 'Fractured';
    const prob   = Math.round((data.fracture_probability || 0) * 100);

    // ── Fracture Probability card ──────────────────────────────────────────
    const probVal = document.getElementById('prob-val');
    probVal.textContent = prob + '%';
    probVal.className   = 'card-value ' + (isFrac ? 'alert-red' : 'text-teal');

    const probSub = document.getElementById('prob-sub');
    probSub.textContent = isFrac ? 'HIGH RISK — FRACTURED' : 'LOW RISK — NON-FRACTURED';
    probSub.className   = 'card-subtitle ' + (isFrac ? 'alert-red' : 'text-teal');

    // ── YOLO model card ────────────────────────────────────────────────────
    const yoloVal = document.getElementById('yolo-val');
    const yoloSub = document.getElementById('yolo-sub');
    if (data.yolo_confidence != null) {
        yoloVal.textContent = data.yolo_confidence.toFixed(2);
        yoloVal.className   = 'model-card-value alert-red';
        if ((data.mode === 'GEL' || data.mode === 'GEL-DEGRADED') && data.gel_gate_passed === false) {
            yoloSub.textContent = 'Y1B · Gate ✗';
        } else {
            yoloSub.textContent = 'Y1B · BOX DETECTED';
        }
    } else if (data.mode === 'CLASSIFIER-ONLY') {
        yoloVal.textContent = '—';
        yoloVal.className   = 'model-card-value';
        yoloSub.textContent = 'Y1B · skipped';
    } else {
        yoloVal.textContent = '—';
        yoloVal.className   = 'model-card-value text-teal';
        yoloSub.textContent = 'Y1B · NO DETECTION';
    }

    // ── ResNet-18 model card ───────────────────────────────────────────────
    const resnetVal = document.getElementById('resnet-val');
    const resnetSub = document.getElementById('resnet-sub');
    if (data.resnet_probability != null) {
        const rp = data.resnet_probability;
        resnetVal.textContent = rp.toFixed(2);
        resnetVal.className   = 'model-card-value ' + (rp >= 0.525 ? 'alert-red' : 'text-teal');
        resnetSub.textContent = (data.mode === 'GEL' || data.mode === 'GEL-DEGRADED') ? 'E6 · GEL' : 'E6 · thr 0.525';
    } else if (data.mode === 'YOLO-ONLY') {
        resnetVal.textContent = '—';
        resnetVal.className   = 'model-card-value';
        resnetSub.textContent = 'E6 · skipped';
    } else {
        resnetVal.textContent = '—';
        resnetVal.className   = 'model-card-value';
        resnetSub.textContent = 'E6 · not loaded';
    }

    // ── DenseNet-169 model card ────────────────────────────────────────────
    const densenetVal = document.getElementById('densenet-val');
    const densenetSub = document.getElementById('densenet-sub');
    if (data.densenet_probability != null) {
        const dp = data.densenet_probability;
        densenetVal.textContent = dp.toFixed(2);
        densenetVal.className   = 'model-card-value ' + (isFrac ? 'alert-red' : 'text-teal');
        densenetSub.textContent = (data.mode === 'GEL' || data.mode === 'GEL-DEGRADED') ? 'D1 · GEL' : 'D1 · thr 0.175';
    } else if (data.mode === 'YOLO-ONLY') {
        densenetVal.textContent = '—';
        densenetVal.className   = 'model-card-value';
        densenetSub.textContent = 'D1 · skipped';
    } else {
        densenetVal.textContent = '—';
        densenetVal.className   = 'model-card-value';
        densenetSub.textContent = 'D1 · not loaded';
    }

    // ── EfficientNet-B3 model card ─────────────────────────────────────────
    const efficientnetVal = document.getElementById('efficientnet-val');
    const efficientnetSub = document.getElementById('efficientnet-sub');
    if (data.efficientnet_probability != null) {
        const ep = data.efficientnet_probability;
        efficientnetVal.textContent = ep.toFixed(2);
        efficientnetVal.className   = 'model-card-value ' + (isFrac ? 'alert-red' : 'text-teal');
        efficientnetSub.textContent = (data.mode === 'GEL' || data.mode === 'GEL-DEGRADED') ? 'F1 · GEL' : 'F1 · thr 0.525';
    } else if (data.mode === 'YOLO-ONLY') {
        efficientnetVal.textContent = '—';
        efficientnetVal.className   = 'model-card-value';
        efficientnetSub.textContent = 'F1 · skipped';
    } else {
        efficientnetVal.textContent = '—';
        efficientnetVal.className   = 'model-card-value';
        efficientnetSub.textContent = 'F1 · not loaded';
    }

    // ── Status banner ──────────────────────────────────────────────────────
    const banner = document.getElementById('status-banner');
    banner.className = 'status-banner ' + (isFrac ? 'status-fractured' : 'status-ok');
    document.getElementById('status-dot').className  = 'status-dot ' + (isFrac ? 'dot-red' : 'dot-teal');
    const _modeLabels  = { 'YOLO-ONLY': 'YOLO ONLY', 'CLASSIFIER-ONLY': 'CLASSIFIER ONLY',
                           'GEL': 'GEL ENSEMBLE',   'GEL-DEGRADED': 'GEL DEGRADED' };
    const _modelLabels = { 'YOLO-ONLY': 'Y1B',      'CLASSIFIER-ONLY': 'E4a · D1 · F1',
                           'GEL': 'Y1B · E4a · D1 · F1', 'GEL-DEGRADED': 'partial models' };
    document.getElementById('status-text').textContent  = _modeLabels[data.mode]  || data.mode;
    document.getElementById('status-model').textContent = _modelLabels[data.mode] || '';

    showState('result');
    _updateOverlay(_currentOverlay);
}

// ─── Overlay image swap ─────────────────────────────────────────────────────
function _updateOverlay(mode) {
    if (!_resultData) return;
    const img   = document.getElementById('result-img');
    const badge = document.getElementById('img-badge');

    if (mode === 'grad' && _resultData.gradcam_image) {
        img.src = _resultData.gradcam_image;
        badge.textContent = 'GradCAM · DenseNet-169 · denseblock4';
    } else if (mode === 'box' && _resultData.xray_with_box) {
        img.src = _resultData.xray_with_box;
        const conf = _resultData.yolo_confidence != null
            ? (_resultData.yolo_confidence * 100).toFixed(0) + '%'
            : '—';
        badge.textContent = `YOLO · Y1B · conf ${conf}`;
    } else {
        img.src = _resultData.gradcam_image || _resultData.xray_with_box || '';
        badge.textContent = mode === 'box' ? 'No YOLO detection' : 'GradCAM unavailable';
    }
}

// ─── UI state helpers ──────────────────────────────────────────────────────
function showState(state) {
    document.getElementById('state-empty').classList.toggle('hidden', state !== 'empty');
    document.getElementById('state-loading').classList.toggle('hidden', state !== 'loading');
    document.getElementById('state-result').classList.toggle('hidden', state !== 'result');
}

function resetMetrics() {
    // Fracture probability card
    const probVal = document.getElementById('prob-val');
    probVal.textContent = '—';
    probVal.className   = 'card-value';
    const probSub = document.getElementById('prob-sub');
    probSub.textContent = 'awaiting image';
    probSub.className   = 'card-subtitle';

    // Model cards
    [
        ['yolo-val',          'model-card-value', 'yolo-sub',          'Y1B'],
        ['resnet-val',        'model-card-value', 'resnet-sub',        'E6'],
        ['densenet-val',      'model-card-value', 'densenet-sub',      'D1'],
        ['efficientnet-val',  'model-card-value', 'efficientnet-sub',  'F1'],
    ].forEach(([valId, valClass, subId, subText]) => {
        const v = document.getElementById(valId);
        v.textContent = '—';
        v.className   = valClass;
        document.getElementById(subId).textContent = subText;
    });

    // Status banner
    const banner = document.getElementById('status-banner');
    banner.className = 'status-banner';
    document.getElementById('status-dot').className   = 'status-dot';
    document.getElementById('status-text').textContent = 'awaiting prediction';
    document.getElementById('status-model').textContent = '';
}

function showError(msg) {
    showState('empty');
    const banner = document.getElementById('status-banner');
    banner.className = 'status-banner status-error';
    document.getElementById('status-dot').className   = 'status-dot dot-amber';
    document.getElementById('status-text').textContent = 'ERROR';
    document.getElementById('status-model').textContent = msg;
}

function resetUI() {
    _resultData = null;
    showState('empty');
    resetMetrics();
    document.getElementById('view-box').checked = true;
    _currentOverlay = 'box';
    document.getElementById('img-id-badge').textContent = '';
}

// ─── Send Review ───────────────────────────────────────────────────────────
document.getElementById('send-review-btn').addEventListener('click', () => {
    if (!_resultData || !_currentFilename) return;
    const btn = document.getElementById('send-review-btn');
    btn.textContent = 'Sending…';
    btn.disabled = true;

    fetch(`${API_URL}/send-review`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            image_id:               _currentFilename,
            gel_probability:        _resultData.fracture_probability ?? '',
            gel_label:              _resultData.label ?? '',
            resnet_probability:     _resultData.resnet_probability ?? '',
            densenet_probability:   _resultData.densenet_probability ?? '',
            efficientnet_probability: _resultData.efficientnet_probability ?? '',
        }),
    })
    .then(r => r.json())
    .then(d => {
        btn.textContent = d.error ? 'Already sent' : 'Sent ✓';
        setTimeout(() => { btn.textContent = 'Send Review'; btn.disabled = false; }, 2000);
    })
    .catch(() => {
        btn.textContent = 'Error';
        setTimeout(() => { btn.textContent = 'Send Review'; btn.disabled = false; }, 2000);
    });
});

// ─── Expert Review queue loader ────────────────────────────────────────────
function _loadReviewQueue() {
    fetch(`${API_URL}/review-queue`)
        .then(r => r.json())
        .then(rows => {
            const queue   = document.getElementById('review-queue');
            const counter = document.getElementById('review-count');
            queue.innerHTML = '';
            const pending = rows.filter(r => r.status === 'pending').length;
            counter.textContent = pending + ' pending';

            rows.forEach(row => {
                const isDone   = row.status === 'diagnosed';
                const gelProb  = parseFloat(row.gel_probability  || 0);
                const rProb    = parseFloat(row.resnet_probability || 0);
                const dProb    = parseFloat(row.densenet_probability || 0);
                const eProb    = parseFloat(row.efficientnet_probability || 0);
                const isFrac   = row.gel_label === 'Fractured';

                const trueFrac  = row.true_label === 'Fractured';
                const gelFrac   = isFrac;
                const rFrac     = rProb >= 0.525;
                const dFrac     = dProb >= 0.175;
                const eFrac     = eProb >= 0.525;

                const trueClass = trueFrac ? 'condition-fractured' : 'condition-nonfractured';
                const trueLabel = trueFrac ? 'FRACTURED' : 'NON-FRACTURED';
                const gelClass  = gelFrac  ? 'condition-fractured' : 'condition-nonfractured';
                const gelLabel  = gelFrac  ? 'FRACTURED' : 'NON-FRACTURED';

                const chipR = `<span class="prob-chip ${rFrac ? 'prob-fracture' : ''}">ResNet ${rProb.toFixed(2)} · ${rFrac ? 'FRAC' : 'NON-FRAC'}</span>`;
                const chipD = `<span class="prob-chip ${dFrac ? 'prob-fracture' : ''}">DenseNet ${dProb.toFixed(2)} · ${dFrac ? 'FRAC' : 'NON-FRAC'}</span>`;
                const chipE = `<span class="prob-chip ${eFrac ? 'prob-fracture' : ''}">EfficientNet ${eProb.toFixed(2)} · ${eFrac ? 'FRAC' : 'NON-FRAC'}</span>`;
                const chipG = `<span class="prob-chip ${gelFrac ? 'prob-fracture' : ''}">GEL ${gelProb.toFixed(2)} · ${gelFrac ? 'FRAC' : 'NON-FRAC'}</span>`;

                const actionBtn = isDone
                    ? `<button class="review-action-btn btn-diagnosed" disabled>Diagnosed</button>`
                    : `<button class="review-action-btn btn-review" onclick="openDiagnose(this,'${row.image_id}')">Review</button>`;

                queue.insertAdjacentHTML('beforeend', `
                    <div class="review-row ${isDone ? 'review-row-done' : ''}" data-rowid="${row.image_id}">
                        <div class="review-col-id"><span class="review-id-text">${row.image_id}</span></div>
                        <div class="review-col-thumb"><div class="review-thumb-box"><img class="review-thumb-img" src="${API_URL}/review/images/${row.image_id}" alt=""></div></div>
                        <div class="review-col-condition"><span class="condition-badge ${trueClass}">${trueLabel}</span></div>
                        <div class="review-col-condition"><span class="condition-badge ${gelClass}">${gelLabel}</span></div>
                        <div class="review-col-prob">${chipG}${chipR}${chipD}${chipE}</div>
                        <div class="review-col-action">${actionBtn}</div>
                    </div>`);
            });
        })
        .catch(() => {});
}

// ─── Health check on load — populate Config device field ──────────────────
fetch(`${API_URL}/health`)
    .then(r => r.json())
    .then(d => {
        if (d.status === 'ok') {
            const el = document.getElementById('cfg-device');
            if (el) el.textContent = d.device || 'connected';
        }
    })
    .catch(() => {
        // Server not running — silent until user tries to predict
    });
