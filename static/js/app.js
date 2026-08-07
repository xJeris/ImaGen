/**
 * ImaGen v2.0 — Main UI Logic
 *
 * Handles tab switching, form state, generation calls, and result display.
 */

// ── DOM references ──────────────────────────────────────────

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

// ── App state ───────────────────────────────────────────────

const state = {
  mode: 'image',        // top-level mode
  subMode: 't2i',       // active sub-mode
  generating: false,
  batchImages: [],       // base64 strings
  batchIndex: 0,
  currentImage: null,    // base64 of displayed image
  lastSeed: -1,
  // I2I state
  i2iSourceFile: null,   // File object for img2img source
  // Inpaint state
  inpaintSourceFile: null, // File object for inpaint source
  inpaintTool: 'brush',
  inpaintBrushSize: 30,
  inpaintDrawing: false,
  // Video state
  videoGenerating: false,
  // Animate state
  animGenerating: false,
  animSourceFile: null,   // File object for animation source
};

// ── Initialization ──────────────────────────────────────────

document.addEventListener('DOMContentLoaded', async () => {
  // Connect WebSocket
  API.connectWebSocket();
  API.onProgress(handleWSMessage);

  // Wire up accordion toggles
  $$('.accordion-header').forEach(h => {
    h.addEventListener('click', () => toggleAccordion(h));
  });

  // Wire up hires fix toggle
  const hiresCheck = $('#hires-check');
  if (hiresCheck) {
    hiresCheck.addEventListener('change', () => {
      $('#hires-settings').style.display = hiresCheck.checked ? 'block' : 'none';
    });
  }

  // Wire up all slider value displays
  $$('.slider-row input[type="range"]').forEach(slider => {
    slider.addEventListener('input', () => {
      slider.nextElementSibling.textContent = slider.value;
    });
  });

  // Wire up brush size slider separately (not in a .slider-row)
  const brushSize = $('#brush-size');
  if (brushSize) {
    brushSize.addEventListener('input', () => {
      state.inpaintBrushSize = parseInt(brushSize.value);
      $('#brush-size-value').textContent = brushSize.value;
    });
  }

  // Wire up seed dice buttons
  $$('.btn-dice').forEach(btn => {
    btn.addEventListener('click', () => {
      const input = btn.parentElement.querySelector('.form-number');
      if (input) input.value = -1;
    });
  });

  // Load initial state from server
  try {
    const [status, archData] = await Promise.all([
      API.getStatus(),
      API.getArchitectures(),
    ]);

    updateVRAM(status.vram);

    // Populate architecture dropdowns (T2I, I2I, Inpaint)
    if (archData.architectures) {
      const archs = archData.architectures;
      const i2iArchs = archData.i2i_architectures || archs;
      populateSelect('#arch-select', archs);
      populateSelect('#i2i-arch-select', i2iArchs);
      populateSelect('#inp-arch-select', i2iArchs);
      if (status.architecture) {
        $('#arch-select').value = status.architecture;
        if (i2iArchs.includes(status.architecture)) {
          $('#i2i-arch-select').value = status.architecture;
          $('#inp-arch-select').value = status.architecture;
        }
      }
    }

    // Load models, loras, vaes, upscalers, schedulers
    await refreshDropdowns();

    // Apply defaults for current architecture
    if (archData.defaults && archData.defaults[status.architecture]) {
      applyDefaults(archData.defaults[status.architecture]);
    }

    // Show prompting guide for current architecture
    if (archData.guides && archData.guides[status.architecture]) {
      updatePromptGuide(archData.guides[status.architecture]);
    }

    // Update status text
    if (status.loaded && status.model) {
      setStatusBar(`Loaded: ${status.model}`);
    }
  } catch (e) {
    showMessage('error', `Failed to connect: ${e.message}`);
    // Retry VRAM update after a delay (server may still be starting)
    setTimeout(refreshVRAM, 3000);
  }

  // Wire architecture change (T2I)
  const archSelect = $('#arch-select');
  if (archSelect) {
    archSelect.addEventListener('change', async () => {
      try {
        showMessage('info', `Switching to ${archSelect.value}...`);
        const result = await API.switchArchitecture(archSelect.value);
        // Sync all three arch dropdowns
        syncArchDropdowns(archSelect.value);
        populateSelectAll('.model-select-sync', result.models, '(none)');
        populateSelectAll('.lora1-select-sync', result.loras);
        populateSelectAll('.lora2-select-sync', result.loras);
        populateSelectAll('.vae-select-sync', result.vaes);
        if (result.schedulers) {
          populateSelectAll('.sampler-select-sync', result.schedulers);
        }
        if (result.defaults) {
          applyDefaults(result.defaults);
        }
        if (result.guide) {
          updatePromptGuide(result.guide);
        }
        showMessage('success', result.status);
      } catch (e) {
        showMessage('error', e.message);
      }
    });
  }

  // Wire architecture change (I2I and Inpaint — sync all)
  ['#i2i-arch-select', '#inp-arch-select'].forEach(sel => {
    const el = $(sel);
    if (el) {
      el.addEventListener('change', async () => {
        try {
          showMessage('info', `Switching to ${el.value}...`);
          const result = await API.switchArchitecture(el.value);
          syncArchDropdowns(el.value);
          populateSelectAll('.model-select-sync', result.models, '(none)');
          populateSelectAll('.lora1-select-sync', result.loras);
          populateSelectAll('.lora2-select-sync', result.loras);
          populateSelectAll('.vae-select-sync', result.vaes);
          if (result.schedulers) {
            populateSelectAll('.sampler-select-sync', result.schedulers);
          }
          if (result.defaults) {
            applyDefaults(result.defaults);
          }
          if (result.guide) {
            updatePromptGuide(result.guide);
          }
          showMessage('success', result.status);
        } catch (e) {
          showMessage('error', e.message);
        }
      });
    }
  });

  // Wire model change (any model dropdown)
  $$('.model-select-sync').forEach(modelSelect => {
    modelSelect.addEventListener('change', async () => {
      const name = modelSelect.value;
      if (!name || name === '(none)') return;
      try {
        showMessage('info', `Loading ${name}...`);
        const result = await API.loadModel(name);
        if (result.status === 'confirm_download') {
          showMessage('warning', `Download required: ${result.message}. Select the model again to confirm.`);
          return;
        }
        if (result.loras) {
          populateSelectAll('.lora1-select-sync', result.loras);
          populateSelectAll('.lora2-select-sync', result.loras);
        }
        if (result.vaes) populateSelectAll('.vae-select-sync', result.vaes);
        // Sync model selection across all dropdowns
        $$('.model-select-sync').forEach(s => s.value = name);
        showMessage('success', result.status);
        setStatusBar(result.status);
      } catch (e) {
        showMessage('error', e.message);
      }
    });
  });

  // Wire VAE change (any VAE dropdown)
  $$('.vae-select-sync').forEach(vaeSelect => {
    vaeSelect.addEventListener('change', async () => {
      try {
        const result = await API.loadVae(vaeSelect.value);
        // Sync all VAE dropdowns
        $$('.vae-select-sync').forEach(s => s.value = vaeSelect.value);
        showMessage('success', result.status);
      } catch (e) {
        showMessage('error', e.message);
      }
    });
  });

  // Wire LoRA trigger word display (T2I only for now)
  ['#lora1-select', '#lora2-select'].forEach((sel, i) => {
    const el = $(sel);
    if (el) {
      el.addEventListener('change', async () => {
        const triggerId = `#lora${i + 1}-triggers`;
        const triggerEl = $(triggerId);
        if (!triggerEl) return;
        if (el.value === 'None') {
          triggerEl.innerHTML = '';
          return;
        }
        try {
          const data = await API.getLoraTriggers(el.value);
          if (data.triggers && data.triggers.length) {
            triggerEl.innerHTML = 'Trigger words: ' +
              data.triggers.map(w => `<code>${w}</code>`).join(', ');
          } else {
            triggerEl.innerHTML = '';
          }
        } catch (e) {
          triggerEl.innerHTML = '';
        }
      });
    }
  });

  // Wire T2I generate button
  const genBtn = $('#btn-generate');
  if (genBtn) {
    genBtn.addEventListener('click', () => runGeneration());
  }

  // Wire T2I stop button
  const stopBtn = $('#btn-stop');
  if (stopBtn) {
    stopBtn.addEventListener('click', async () => {
      try { await API.interrupt(); showMessage('info', 'Stopping...'); } catch (e) { showMessage('error', e.message); }
    });
  }

  // Wire T2I save buttons
  const saveBtn = $('#btn-save');
  if (saveBtn) {
    saveBtn.addEventListener('click', async () => {
      try {
        const saveHistory = $('#save-history')?.checked || false;
        const grid = document.querySelector('.batch-grid');
        const checkedCells = grid && grid.style.display !== 'none'
          ? grid.querySelectorAll('.batch-cell:has(.batch-check:checked)')
          : null;

        if (checkedCells && checkedCells.length > 0) {
          let savedCount = 0;
          for (const cell of checkedCells) {
            const idx = parseInt(cell.dataset.index);
            await API.saveImage('t2i', saveHistory, idx);
            savedCount++;
          }
          showMessage('success', `Saved ${savedCount} image${savedCount > 1 ? 's' : ''}`);
        } else {
          const result = await API.saveImage('t2i', saveHistory, state.batchIndex);
          showMessage('success', `Saved: ${result.path}`);
        }
      } catch (e) {
        showMessage('error', e.message);
      }
    });
  }

  const saveAllBtn = $('#btn-save-all');
  if (saveAllBtn) {
    saveAllBtn.addEventListener('click', async () => {
      try {
        const saveHistory = $('#save-history')?.checked || false;
        const result = await API.saveAll(saveHistory);
        showMessage('success', result.status);
      } catch (e) {
        showMessage('error', e.message);
      }
    });
  }

  // Wire I2I generate button
  const genI2iBtn = $('#btn-generate-i2i');
  if (genI2iBtn) {
    genI2iBtn.addEventListener('click', () => runImg2Img());
  }

  // Wire I2I stop button
  const stopI2iBtn = $('#btn-stop-i2i');
  if (stopI2iBtn) {
    stopI2iBtn.addEventListener('click', async () => {
      try { await API.interrupt(); showMessage('info', 'Stopping...'); } catch (e) { showMessage('error', e.message); }
    });
  }

  // Wire I2I save button
  const saveI2iBtn = $('#btn-save-i2i');
  if (saveI2iBtn) {
    saveI2iBtn.addEventListener('click', async () => {
      try {
        const saveHistory = $('#i2i-save-history')?.checked || false;
        const result = await API.saveImage('i2i', saveHistory);
        showMessage('success', `Saved: ${result.path}`);
      } catch (e) {
        showMessage('error', e.message);
      }
    });
  }

  // Wire Inpaint generate button
  const genInpBtn = $('#btn-generate-inpaint');
  if (genInpBtn) {
    genInpBtn.addEventListener('click', () => runInpaint());
  }

  // Wire Inpaint stop button
  const stopInpBtn = $('#btn-stop-inpaint');
  if (stopInpBtn) {
    stopInpBtn.addEventListener('click', async () => {
      try { await API.interrupt(); showMessage('info', 'Stopping...'); } catch (e) { showMessage('error', e.message); }
    });
  }

  // Wire Inpaint save button
  const saveInpBtn = $('#btn-save-inpaint');
  if (saveInpBtn) {
    saveInpBtn.addEventListener('click', async () => {
      try {
        const saveHistory = $('#inp-save-history')?.checked || false;
        const result = await API.saveImage('i2i', saveHistory);
        showMessage('success', `Saved: ${result.path}`);
      } catch (e) {
        showMessage('error', e.message);
      }
    });
  }

  // Wire shutdown button
  const shutdownBtn = $('.shutdown-btn');
  if (shutdownBtn) {
    shutdownBtn.addEventListener('click', async () => {
      if (!confirm('Shut down ImaGen?')) return;
      try {
        await API.shutdown();
        showMessage('info', 'Shutting down...');
      } catch (e) {
        // Expected — server is shutting down
      }
    });
  }

  // ── I2I file upload ──
  initImageUpload('i2i');

  // ── Inpaint file upload + canvas ──
  initImageUpload('inpaint');
  initInpaintCanvas();

  // ══════════════════════════════════════════════════════════
  // VIDEO TAB WIRING
  // ══════════════════════════════════════════════════════════

  // Wire video architecture change
  const videoArchSelect = $('#video-arch-select');
  if (videoArchSelect) {
    videoArchSelect.addEventListener('change', async () => {
      try {
        showMessage('info', `Switching video arch to ${videoArchSelect.value}...`);
        const result = await API.switchVideoArchitecture(videoArchSelect.value);
        populateSelect('#video-model-select', result.models, '(none)');
        if (result.defaults) {
          applyVideoDefaults(result.defaults);
        }
        showMessage('success', result.status);
        try { const s = await API.getStatus(); updateVRAM(s.vram); } catch (_) {}
      } catch (e) {
        showMessage('error', e.message);
      }
    });
  }

  // Wire video model change
  const videoModelSelect = $('#video-model-select');
  if (videoModelSelect) {
    videoModelSelect.addEventListener('change', async () => {
      const name = videoModelSelect.value;
      if (!name || name === '(none)') return;
      try {
        showMessage('info', `Loading video model: ${name}...`);
        const result = await API.loadVideoModel(name);
        showMessage('success', result.status);
        updateVideoVramEstimate();
        try { const s = await API.getStatus(); updateVRAM(s.vram); } catch (_) {}
      } catch (e) {
        showMessage('error', e.message);
      }
    });
  }

  // Wire video generate button
  const genVideoBtn = $('#btn-generate-video');
  if (genVideoBtn) {
    genVideoBtn.addEventListener('click', () => runVideoGeneration());
  }

  // Wire video stop button
  const stopVideoBtn = $('#btn-stop-video');
  if (stopVideoBtn) {
    stopVideoBtn.addEventListener('click', async () => {
      try { await API.interruptVideo(); showMessage('info', 'Stopping...'); } catch (e) { showMessage('error', e.message); }
    });
  }

  // Wire video save button
  const saveVideoBtn = $('#btn-save-video');
  if (saveVideoBtn) {
    saveVideoBtn.addEventListener('click', async () => {
      try {
        const result = await API.saveVideo();
        showMessage('success', `Saved: ${result.path}`);
      } catch (e) {
        showMessage('error', e.message);
      }
    });
  }

  // Wire video VRAM estimate updates on duration/fps change
  const videoDuration = $('#video-duration');
  const videoFps = $('#video-fps');
  if (videoDuration) videoDuration.addEventListener('input', () => updateVideoVramEstimate());
  if (videoFps) videoFps.addEventListener('input', () => updateVideoVramEstimate());

  // Load video models on first switch to video tab
  loadVideoModels();

  // ══════════════════════════════════════════════════════════
  // ANIMATE TAB WIRING
  // ══════════════════════════════════════════════════════════

  initAnimSourceUpload();

  // Wire animate model load button
  const loadAnimBtn = $('#btn-load-anim');
  if (loadAnimBtn) {
    loadAnimBtn.addEventListener('click', () => loadAnimateDiffModels());
  }

  // Wire animate button
  const animBtn = $('#btn-animate');
  if (animBtn) {
    animBtn.addEventListener('click', () => runAnimation());
  }

  // Wire animate stop button
  const stopAnimBtn = $('#btn-stop-anim');
  if (stopAnimBtn) {
    stopAnimBtn.addEventListener('click', async () => {
      try { await API.interruptAnimation(); showMessage('info', 'Stopping...'); } catch (e) { showMessage('error', e.message); }
    });
  }

  // Wire animate save button
  const saveAnimBtn = $('#btn-save-anim');
  if (saveAnimBtn) {
    saveAnimBtn.addEventListener('click', async () => {
      try {
        const result = await API.saveAnimation();
        showMessage('success', `Saved: ${result.path}`);
      } catch (e) {
        showMessage('error', e.message);
      }
    });
  }

  // Wire animate VRAM estimate updates
  const animFrames = $('#anim-frames');
  const animFps = $('#anim-fps');
  if (animFrames) animFrames.addEventListener('input', () => updateAnimVramEstimate());
  if (animFps) animFps.addEventListener('input', () => updateAnimVramEstimate());

  // Load animate model lists
  loadAnimateModelLists();

  // ── Model Browser event wiring ──
  const btnSearch = $('#btn-search');
  if (btnSearch) btnSearch.addEventListener('click', () => searchBrowser());

  const browserQuery = $('#browser-query');
  if (browserQuery) browserQuery.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') searchBrowser();
  });

  const btnDownload = $('#btn-download');
  if (btnDownload) btnDownload.addEventListener('click', () => downloadBrowserModel());

  const btnBrowserPrev = $('#btn-browser-prev');
  if (btnBrowserPrev) btnBrowserPrev.addEventListener('click', () => browserPrevPage());

  const btnBrowserNext = $('#btn-browser-next');
  if (btnBrowserNext) btnBrowserNext.addEventListener('click', () => browserNextPage());

  const btnSaveApikey = $('#btn-save-apikey');
  if (btnSaveApikey) btnSaveApikey.addEventListener('click', async () => {
    const key = $('#civitai-api-key')?.value || '';
    try {
      await API.saveCivitaiKey(key);
      showMessage('success', 'API key saved');
    } catch (e) {
      showMessage('error', e.message);
    }
  });

  // Load API key on startup
  API.getCivitaiKey().then(data => {
    const keyInput = $('#civitai-api-key');
    if (keyInput && data.key) keyInput.value = data.key;
  }).catch(() => {});

  // ── Preview Files event wiring ──
  const previewFilter = $('#preview-filter-type');
  const previewSort = $('#preview-sort');
  if (previewFilter) previewFilter.addEventListener('change', () => loadPreviewFiles());
  if (previewSort) previewSort.addEventListener('change', () => loadPreviewFiles());

  const btnRefresh = $('#btn-preview-refresh');
  if (btnRefresh) btnRefresh.addEventListener('click', () => loadPreviewFiles());

  const btnSelectAll = $('#btn-select-all');
  if (btnSelectAll) btnSelectAll.addEventListener('click', () => {
    $$('#canvas-preview .card-check').forEach(cb => cb.checked = true);
    updateBatchDeleteCount();
  });

  const btnDeselectAll = $('#btn-deselect-all');
  if (btnDeselectAll) btnDeselectAll.addEventListener('click', () => {
    $$('#canvas-preview .card-check').forEach(cb => cb.checked = false);
    updateBatchDeleteCount();
  });

  const btnBatchDelete = $('#btn-batch-delete');
  if (btnBatchDelete) btnBatchDelete.addEventListener('click', async () => {
    const checked = $$('#canvas-preview .card-check:checked');
    if (!checked.length) return;
    const filenames = checked.map(cb => cb.closest('.preview-card').querySelector('.card-name').textContent);
    if (!confirm(`Delete ${filenames.length} file(s)?`)) return;
    try {
      await API.deleteOutputsBatch(filenames);
      showMessage('success', `Deleted ${filenames.length} files`);
      loadPreviewFiles();
      // Reset detail panel
      const detail = $('#file-detail');
      const placeholder = $('#file-detail-placeholder');
      if (detail) detail.classList.remove('visible');
      if (placeholder) placeholder.style.display = 'block';
    } catch (e) {
      showMessage('error', e.message);
    }
  });

  // ── LoRA Training event wiring ──
  const btnTrain = $('#btn-train');
  if (btnTrain) btnTrain.addEventListener('click', () => startTraining());

  const btnStopTrain = $('#btn-stop-train');
  if (btnStopTrain) btnStopTrain.addEventListener('click', () => stopTraining());

  // Training slider sync
  const trainSteps = $('#train-steps');
  if (trainSteps) trainSteps.addEventListener('input', () => {
    trainSteps.nextElementSibling.textContent = trainSteps.value;
  });

  const trainLr = $('#train-lr');
  if (trainLr) trainLr.addEventListener('input', () => {
    trainLr.nextElementSibling.textContent = `1e${trainLr.value}`;
  });

  const trainRank = $('#train-rank');
  if (trainRank) trainRank.addEventListener('input', () => {
    trainRank.nextElementSibling.textContent = trainRank.value;
  });

  // ── Profile event wiring ──
  const btnProfileLoad = $('#btn-profile-load');
  if (btnProfileLoad) btnProfileLoad.addEventListener('click', () => loadSelectedProfile());

  const btnProfileSave = $('#btn-profile-save');
  if (btnProfileSave) btnProfileSave.addEventListener('click', () => showSaveProfileInput());

  const btnProfileSaveConfirm = $('#btn-profile-save-confirm');
  if (btnProfileSaveConfirm) btnProfileSaveConfirm.addEventListener('click', () => confirmSaveProfile());

  const profileSaveName = $('#profile-save-name');
  if (profileSaveName) profileSaveName.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') confirmSaveProfile();
  });

  const btnProfileDelete = $('#btn-profile-delete');
  if (btnProfileDelete) btnProfileDelete.addEventListener('click', () => deleteSelectedProfile());

  // Load profile list on startup and auto-fill default prompts
  loadProfiles(true);
});

// ── Mode switching (top nav) ────────────────────────────────

function switchMode(mode) {
  state.mode = mode;

  // Update top nav tabs
  $$('.top-nav .nav-tab').forEach(t => {
    t.classList.toggle('active', t.dataset.mode === mode);
  });

  // Show/hide bottom nav groups
  const allModes = ['image', 'video', 'browser', 'preview', 'train'];
  allModes.forEach(m => {
    document.getElementById('bottom-' + m).classList.toggle('panel-hidden', m !== mode);
  });

  // Activate the default sub-tab for this mode
  const bottomGroup = document.getElementById('bottom-' + mode);
  const activeSubTab = bottomGroup.querySelector('.sub-tab.active');
  if (activeSubTab) {
    switchSubMode(activeSubTab.dataset.sub, activeSubTab);
  }

  // Update online/offline badge
  const badge = $('#net-badge');
  const needsOnline = (mode === 'browser');
  badge.textContent = needsOnline ? 'Online' : 'Offline';
  badge.classList.toggle('online', needsOnline);
  badge.classList.toggle('offline', !needsOnline);
}

// ── Sub-mode switching (bottom nav) ─────────────────────────

function switchSubMode(sub, el) {
  state.subMode = sub;

  // Update active state within parent group
  const group = el.parentElement;
  group.querySelectorAll('.sub-tab').forEach(t => t.classList.remove('active'));
  el.classList.add('active');

  // Canvas panel mapping
  const canvasPanels = {
    t2i: 'canvas-image', i2i: 'canvas-i2i', inpaint: 'canvas-inpaint',
    t2v: 'canvas-video', animate: 'canvas-animate',
    browse: 'canvas-browser', preview: 'canvas-preview', train: 'canvas-training',
  };
  // Sidebar panel mapping
  const sidebarPanels = {
    t2i: 'sidebar-image', i2i: 'sidebar-i2i', inpaint: 'sidebar-inpaint',
    t2v: 'sidebar-video', animate: 'sidebar-animate',
    browse: 'sidebar-browser', preview: 'sidebar-preview', train: 'sidebar-training',
  };

  const allCanvas = ['canvas-image', 'canvas-i2i', 'canvas-inpaint', 'canvas-video', 'canvas-animate', 'canvas-browser', 'canvas-preview', 'canvas-training'];
  const allSidebar = ['sidebar-image', 'sidebar-i2i', 'sidebar-inpaint', 'sidebar-video', 'sidebar-animate', 'sidebar-browser', 'sidebar-preview', 'sidebar-training'];

  allCanvas.forEach(id => {
    const el2 = document.getElementById(id);
    if (el2) el2.classList.toggle('panel-hidden', id !== canvasPanels[sub]);
  });

  // Show/hide the image-area wrapper (contains t2i, i2i, inpaint canvases)
  const imageArea = document.getElementById('canvas-image-area');
  if (imageArea) {
    const imageAreaModes = ['t2i', 'i2i', 'inpaint'];
    imageArea.classList.toggle('panel-hidden', !imageAreaModes.includes(sub));
  }
  allSidebar.forEach(id => {
    const el2 = document.getElementById(id);
    if (el2) el2.classList.toggle('panel-hidden', id !== sidebarPanels[sub]);
  });

  // Load preview files when switching to preview mode
  if (sub === 'preview') {
    loadPreviewFiles();
  }
}

// ── Accordion toggle ────────────────────────────────────────

function toggleAccordion(header) {
  header.classList.toggle('open');
  const body = header.nextElementSibling;
  body.classList.toggle('open');
}

// ── Message bar ─────────────────────────────────────────────

let _messageTimer = null;

function showMessage(type, text, autoDismiss = null) {
  const bar = $('#message-bar');
  if (!bar) return;

  const icons = { info: '\u2139\uFE0F', warning: '\u26A0\uFE0F', error: '\u274C', success: '\u2705' };

  bar.innerHTML = `
    <div class="msg ${type}">
      <span class="msg-icon">${icons[type] || ''}</span>
      <span>${text}</span>
      <button class="msg-close" onclick="dismissMessage()">&times;</button>
    </div>
  `;
  bar.classList.add('visible');

  if (_messageTimer) clearTimeout(_messageTimer);

  // Auto-dismiss logic
  if (autoDismiss === null) {
    autoDismiss = (type === 'info' || type === 'success') ? 8000 : false;
  }
  if (autoDismiss) {
    _messageTimer = setTimeout(() => dismissMessage(), autoDismiss);
  }
}

function dismissMessage() {
  const bar = $('#message-bar');
  if (bar) {
    bar.classList.remove('visible');
    if (_messageTimer) {
      clearTimeout(_messageTimer);
      _messageTimer = null;
    }
  }
}

// ── VRAM display ────────────────────────────────────────────

function updateVRAM(vram) {
  const el = $('.status-text');
  if (!el || !vram || !vram.available) return;
  el.textContent = `VRAM: ${vram.free_gb} / ${vram.total_gb} GB free`;
}

async function refreshVRAM() {
  try {
    const status = await API.getStatus();
    updateVRAM(status.vram);
  } catch (_) {}
}

// Refresh VRAM display every 10 seconds
setInterval(refreshVRAM, 10000);

// ── Status bar ──────────────────────────────────────────────

function setStatusBar(text) {
  const statusSpan = $('.status-bar > span:first-child');
  if (statusSpan) {
    statusSpan.innerHTML = `<span class="status-dot"></span> ${text}`;
  }
}

function setSeedDisplay(seed) {
  const seedSpan = $('.status-bar > span:last-child');
  if (seedSpan) {
    seedSpan.textContent = `Seed: ${seed}`;
  }
}

// ── Dropdown population ─────────────────────────────────────

function populateSelect(selector, items, placeholder = null) {
  const el = $(selector);
  if (!el) return;
  const current = el.value;
  el.innerHTML = '';
  if (placeholder) {
    const opt = document.createElement('option');
    opt.value = '';
    opt.textContent = placeholder;
    el.appendChild(opt);
  }
  (items || []).forEach(item => {
    const opt = document.createElement('option');
    opt.value = item;
    opt.textContent = item;
    el.appendChild(opt);
  });
  // Restore previous selection if it still exists
  if (current && items && items.includes(current)) {
    el.value = current;
  }
}

/** Populate all elements matching a CSS class selector */
function populateSelectAll(selector, items, placeholder = null) {
  $$(selector).forEach(el => {
    const current = el.value;
    el.innerHTML = '';
    if (placeholder) {
      const opt = document.createElement('option');
      opt.value = '';
      opt.textContent = placeholder;
      el.appendChild(opt);
    }
    (items || []).forEach(item => {
      const opt = document.createElement('option');
      opt.value = item;
      opt.textContent = item;
      el.appendChild(opt);
    });
    if (current && items && items.includes(current)) {
      el.value = current;
    }
  });
}

/** Sync all architecture dropdowns to the same value */
function syncArchDropdowns(value) {
  ['#arch-select', '#i2i-arch-select', '#inp-arch-select'].forEach(sel => {
    const el = $(sel);
    if (el) {
      // Only set if the value exists in this dropdown
      const opts = Array.from(el.options).map(o => o.value);
      if (opts.includes(value)) el.value = value;
    }
  });
}

async function refreshDropdowns() {
  try {
    const [models, loras, vaes, upscalers, schedulers] = await Promise.all([
      API.getModels(),
      API.getLoras(),
      API.getVaes(),
      API.getUpscalers(),
      API.getSchedulers(),
    ]);

    populateSelectAll('.model-select-sync', models.models, '(none)');
    if (models.active) {
      $$('.model-select-sync').forEach(s => s.value = models.active);
    }

    populateSelectAll('.lora1-select-sync', loras.loras);
    populateSelectAll('.lora2-select-sync', loras.loras);
    populateSelectAll('.vae-select-sync', vaes.vaes);
    populateSelectAll('.upscaler-select-sync', upscalers.upscalers);
    populateSelectAll('.sampler-select-sync', schedulers.schedulers);

    // Also populate hires upscaler (T2I only)
    populateSelect('#hires-upscaler-select', upscalers.upscalers);
  } catch (e) {
    console.error('Failed to load dropdowns:', e);
  }
}

// ── Apply architecture defaults ─────────────────────────────

function applyDefaults(defaults) {
  if (!defaults) return;

  // Apply to T2I sliders
  const stepsSlider = $('#steps-slider');
  if (stepsSlider && defaults.steps) {
    stepsSlider.value = defaults.steps;
    stepsSlider.nextElementSibling.textContent = defaults.steps;
  }
  const cfgSlider = $('#cfg-slider');
  if (cfgSlider && defaults.guidance_scale !== undefined) {
    cfgSlider.value = defaults.guidance_scale;
    cfgSlider.nextElementSibling.textContent = defaults.guidance_scale;
  }
  const widthInput = $('#width-input');
  if (widthInput && defaults.width) widthInput.value = defaults.width;
  const heightInput = $('#height-input');
  if (heightInput && defaults.height) heightInput.value = defaults.height;
  if (defaults.scheduler) {
    $$('.sampler-select-sync').forEach(s => s.value = defaults.scheduler);
  }

  // Apply to I2I sliders
  const i2iSteps = $('#i2i-steps-slider');
  if (i2iSteps && defaults.steps) {
    i2iSteps.value = defaults.steps;
    i2iSteps.nextElementSibling.textContent = defaults.steps;
  }
  const i2iCfg = $('#i2i-cfg-slider');
  if (i2iCfg && defaults.guidance_scale !== undefined) {
    i2iCfg.value = defaults.guidance_scale;
    i2iCfg.nextElementSibling.textContent = defaults.guidance_scale;
  }

  // Apply to Inpaint sliders
  const inpSteps = $('#inp-steps-slider');
  if (inpSteps && defaults.steps) {
    inpSteps.value = defaults.steps;
    inpSteps.nextElementSibling.textContent = defaults.steps;
  }
  const inpCfg = $('#inp-cfg-slider');
  if (inpCfg && defaults.guidance_scale !== undefined) {
    inpCfg.value = defaults.guidance_scale;
    inpCfg.nextElementSibling.textContent = defaults.guidance_scale;
  }
}

// ── Prompting Guide ─────────────────────────────────────────

function updatePromptGuide(html) {
  const el = $('#prompt-guide');
  if (el) el.innerHTML = html || '';
}

// ══════════════════════════════════════════════════════════════
// IMAGE UPLOAD (shared by I2I and Inpaint)
// ══════════════════════════════════════════════════════════════

function initImageUpload(mode) {
  const dropzone = $(`#${mode}-dropzone`);
  const fileInput = $(`#${mode}-file-input`);
  if (!dropzone || !fileInput) return;

  // Click to upload
  dropzone.addEventListener('click', () => fileInput.click());

  // File selected
  fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) {
      handleImageFile(mode, fileInput.files[0]);
    }
  });

  // Drag events
  dropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropzone.classList.add('drag-over');
  });
  dropzone.addEventListener('dragleave', () => {
    dropzone.classList.remove('drag-over');
  });
  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('drag-over');
    if (e.dataTransfer.files.length > 0) {
      handleImageFile(mode, e.dataTransfer.files[0]);
    }
  });

  // Clear button (I2I only — inpaint uses toolbar)
  if (mode === 'i2i') {
    const clearBtn = $('#i2i-clear-source');
    if (clearBtn) {
      clearBtn.addEventListener('click', () => clearI2ISource());
    }
  }
}

function handleImageFile(mode, file) {
  if (!file.type.startsWith('image/')) {
    showMessage('error', 'Please upload an image file.');
    return;
  }

  if (mode === 'i2i') {
    state.i2iSourceFile = file;
    const preview = $('#i2i-source-preview');
    const dropzone = $('#i2i-dropzone');
    const clearBtn = $('#i2i-clear-source');

    const url = URL.createObjectURL(file);
    preview.src = url;
    preview.style.display = 'block';
    dropzone.style.display = 'none';
    clearBtn.style.display = 'block';
  } else if (mode === 'inpaint') {
    state.inpaintSourceFile = file;
    loadInpaintImage(file);
  }
}

function clearI2ISource() {
  state.i2iSourceFile = null;
  const preview = $('#i2i-source-preview');
  const dropzone = $('#i2i-dropzone');
  const clearBtn = $('#i2i-clear-source');
  const fileInput = $('#i2i-file-input');

  preview.style.display = 'none';
  preview.src = '';
  dropzone.style.display = 'flex';
  clearBtn.style.display = 'none';
  fileInput.value = '';
}

// ══════════════════════════════════════════════════════════════
// INPAINT CANVAS
// ══════════════════════════════════════════════════════════════

function loadInpaintImage(file) {
  const reader = new FileReader();
  reader.onload = (e) => {
    const img = new Image();
    img.onload = () => {
      const wrap = $('#inpaint-canvas-wrap');
      const sourceCanvas = $('#inpaint-source-canvas');
      const maskCanvas = $('#inpaint-mask-canvas');
      const dropzone = $('#inpaint-dropzone');
      const toolbar = $('#inpaint-toolbar');

      // Size canvases to fit within the container while preserving aspect ratio
      const maxW = wrap.clientWidth;
      const maxH = wrap.clientHeight - 50; // leave room for toolbar
      const scale = Math.min(maxW / img.width, maxH / img.height, 1);
      const w = Math.round(img.width * scale);
      const h = Math.round(img.height * scale);

      sourceCanvas.width = img.width;
      sourceCanvas.height = img.height;
      sourceCanvas.style.width = w + 'px';
      sourceCanvas.style.height = h + 'px';

      maskCanvas.width = img.width;
      maskCanvas.height = img.height;
      maskCanvas.style.width = w + 'px';
      maskCanvas.style.height = h + 'px';

      // Center canvases
      const left = Math.round((maxW - w) / 2);
      const top = Math.round((maxH - h) / 2);
      sourceCanvas.style.left = left + 'px';
      sourceCanvas.style.top = top + 'px';
      maskCanvas.style.left = left + 'px';
      maskCanvas.style.top = top + 'px';

      // Draw source image
      const ctx = sourceCanvas.getContext('2d');
      ctx.drawImage(img, 0, 0);

      // Clear mask
      const mctx = maskCanvas.getContext('2d');
      mctx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);

      // Show canvases, hide dropzone
      sourceCanvas.style.display = 'block';
      maskCanvas.style.display = 'block';
      dropzone.style.display = 'none';
      toolbar.style.display = 'flex';
    };
    img.src = e.target.result;
  };
  reader.readAsDataURL(file);
}

function initInpaintCanvas() {
  const maskCanvas = $('#inpaint-mask-canvas');
  if (!maskCanvas) return;

  const getPos = (e) => {
    const rect = maskCanvas.getBoundingClientRect();
    const scaleX = maskCanvas.width / rect.width;
    const scaleY = maskCanvas.height / rect.height;
    return {
      x: (e.clientX - rect.left) * scaleX,
      y: (e.clientY - rect.top) * scaleY,
    };
  };

  maskCanvas.addEventListener('mousedown', (e) => {
    state.inpaintDrawing = true;
    const pos = getPos(e);
    drawOnMask(pos.x, pos.y, true);
  });

  maskCanvas.addEventListener('mousemove', (e) => {
    if (!state.inpaintDrawing) return;
    const pos = getPos(e);
    drawOnMask(pos.x, pos.y, false);
  });

  maskCanvas.addEventListener('mouseup', () => {
    state.inpaintDrawing = false;
  });

  maskCanvas.addEventListener('mouseleave', () => {
    state.inpaintDrawing = false;
  });

  // Toolbar buttons
  const toolbar = $('#inpaint-toolbar');
  if (toolbar) {
    toolbar.querySelectorAll('.tool-btn[data-tool]').forEach(btn => {
      btn.addEventListener('click', () => {
        const tool = btn.dataset.tool;
        if (tool === 'brush' || tool === 'eraser') {
          state.inpaintTool = tool;
          toolbar.querySelectorAll('.tool-btn[data-tool="brush"], .tool-btn[data-tool="eraser"]').forEach(b => {
            b.classList.toggle('active', b.dataset.tool === tool);
          });
        } else if (tool === 'clear-mask') {
          const mctx = maskCanvas.getContext('2d');
          mctx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
        } else if (tool === 'clear-image') {
          clearInpaintSource();
        }
      });
    });
  }
}

let _lastMaskPos = null;

function drawOnMask(x, y, isStart) {
  const maskCanvas = $('#inpaint-mask-canvas');
  if (!maskCanvas) return;

  const ctx = maskCanvas.getContext('2d');
  const size = state.inpaintBrushSize;

  // Scale brush size relative to canvas resolution
  const rect = maskCanvas.getBoundingClientRect();
  const scaledSize = size * (maskCanvas.width / rect.width);

  if (state.inpaintTool === 'eraser') {
    ctx.globalCompositeOperation = 'destination-out';
  } else {
    ctx.globalCompositeOperation = 'source-over';
    ctx.fillStyle = 'rgba(255, 0, 0, 0.5)';
    ctx.strokeStyle = 'rgba(255, 0, 0, 0.5)';
  }

  ctx.lineWidth = scaledSize;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';

  if (isStart || !_lastMaskPos) {
    ctx.beginPath();
    ctx.arc(x, y, scaledSize / 2, 0, Math.PI * 2);
    ctx.fill();
    _lastMaskPos = { x, y };
  } else {
    ctx.beginPath();
    ctx.moveTo(_lastMaskPos.x, _lastMaskPos.y);
    ctx.lineTo(x, y);
    ctx.stroke();
    _lastMaskPos = { x, y };
  }

  ctx.globalCompositeOperation = 'source-over';
}

function clearInpaintSource() {
  state.inpaintSourceFile = null;
  const sourceCanvas = $('#inpaint-source-canvas');
  const maskCanvas = $('#inpaint-mask-canvas');
  const dropzone = $('#inpaint-dropzone');
  const toolbar = $('#inpaint-toolbar');
  const fileInput = $('#inpaint-file-input');

  sourceCanvas.style.display = 'none';
  maskCanvas.style.display = 'none';
  dropzone.style.display = 'flex';
  toolbar.style.display = 'none';
  fileInput.value = '';

  // Clear canvases
  const sctx = sourceCanvas.getContext('2d');
  sctx.clearRect(0, 0, sourceCanvas.width, sourceCanvas.height);
  const mctx = maskCanvas.getContext('2d');
  mctx.clearRect(0, 0, maskCanvas.width, maskCanvas.height);
}

function getMaskBlob() {
  return new Promise((resolve) => {
    const maskCanvas = $('#inpaint-mask-canvas');
    if (!maskCanvas) { resolve(null); return; }

    // Create a grayscale mask: white where painted, black elsewhere
    const w = maskCanvas.width;
    const h = maskCanvas.height;
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = w;
    tempCanvas.height = h;
    const ctx = tempCanvas.getContext('2d');

    // Get mask pixel data
    const maskCtx = maskCanvas.getContext('2d');
    const maskData = maskCtx.getImageData(0, 0, w, h);

    // Create grayscale: any non-transparent pixel becomes white
    const outData = ctx.createImageData(w, h);
    for (let i = 0; i < maskData.data.length; i += 4) {
      const alpha = maskData.data[i + 3];
      const val = alpha > 0 ? 255 : 0;
      outData.data[i] = val;
      outData.data[i + 1] = val;
      outData.data[i + 2] = val;
      outData.data[i + 3] = 255;
    }
    ctx.putImageData(outData, 0, 0);

    tempCanvas.toBlob((blob) => resolve(blob), 'image/png');
  });
}

// ══════════════════════════════════════════════════════════════
// T2I GENERATION
// ══════════════════════════════════════════════════════════════

async function runGeneration() {
  if (state.generating) return;
  dismissMessage();

  const params = {
    positive_prompt: $('#positive-prompt')?.value || '',
    negative_prompt: $('#negative-prompt')?.value || '',
    description: $('#description-prompt')?.value || '',
    steps: parseInt($('#steps-slider')?.value || 30),
    guidance_scale: parseFloat($('#cfg-slider')?.value || 7.5),
    width: parseInt($('#width-input')?.value || 1024),
    height: parseInt($('#height-input')?.value || 1024),
    seed: parseInt($('#seed-input')?.value || -1),
    scheduler: $('#sampler-select')?.value || 'Euler',
    batch_size: parseInt($('#batch-slider')?.value || 1),
    lora1_name: $('#lora1-select')?.value || 'None',
    lora1_weight: parseFloat($('#lora1-weight')?.value || 1.0),
    lora2_name: $('#lora2-select')?.value || 'None',
    lora2_weight: parseFloat($('#lora2-weight')?.value || 1.0),
    upscaler: $('#upscaler-select')?.value || 'None',
    hires_enable: $('#hires-check')?.checked || false,
    hires_upscaler: $('#hires-upscaler-select')?.value || 'Lanczos',
    hires_scale: parseFloat($('#hires-scale')?.value || 1.5),
    hires_denoise: parseFloat($('#hires-denoise')?.value || 0.5),
    hires_steps: parseInt($('#hires-steps')?.value || 15),
  };

  state.generating = true;
  setGeneratingUI(true, 't2i');

  try {
    const result = await API.generate(params);

    if (result.status === 'interrupted') {
      showMessage('info', 'Generation stopped.');
    } else if (result.images && result.images.length > 0) {
      state.batchImages = result.images;
      state.batchIndex = 0;
      state.lastSeed = result.seed;

      if (result.images.length > 1) {
        displayBatchGrid(result.images);
      } else {
        hideBatchGrid();
        displayImage(result.images[0], '#canvas-image');
      }

      setSeedDisplay(result.seed);
      showMessage('success', `Generated. Seed: ${result.seed}`);
    }
  } catch (e) {
    showMessage('error', e.message);
  } finally {
    state.generating = false;
    setGeneratingUI(false, 't2i');
    try {
      const status = await API.getStatus();
      updateVRAM(status.vram);
    } catch (_) {}
  }
}

// ══════════════════════════════════════════════════════════════
// I2I GENERATION
// ══════════════════════════════════════════════════════════════

async function runImg2Img() {
  if (state.generating) return;
  dismissMessage();
  if (!state.i2iSourceFile) {
    showMessage('error', 'Please upload a source image first.');
    return;
  }

  const fd = new FormData();
  fd.append('source_image', state.i2iSourceFile);
  fd.append('positive_prompt', $('#i2i-positive-prompt')?.value || '');
  fd.append('negative_prompt', $('#i2i-negative-prompt')?.value || '');
  fd.append('description', $('#i2i-description-prompt')?.value || '');
  fd.append('strength', $('#i2i-strength')?.value || '0.7');
  fd.append('steps', $('#i2i-steps-slider')?.value || '30');
  fd.append('guidance_scale', $('#i2i-cfg-slider')?.value || '7.5');
  fd.append('seed', $('#i2i-seed-input')?.value || '-1');
  fd.append('scheduler', $('#i2i-sampler-select')?.value || 'Euler');
  fd.append('lora1_name', $('#i2i-lora1-select')?.value || 'None');
  fd.append('lora1_weight', $('#i2i-lora1-weight')?.value || '1.0');
  fd.append('lora2_name', $('#i2i-lora2-select')?.value || 'None');
  fd.append('lora2_weight', $('#i2i-lora2-weight')?.value || '1.0');
  fd.append('upscaler', $('#i2i-upscaler-select')?.value || 'None');

  state.generating = true;
  setGeneratingUI(true, 'i2i');

  try {
    const result = await API.img2img(fd);

    if (result.status === 'interrupted') {
      showMessage('info', 'Generation stopped.');
    } else if (result.images && result.images.length > 0) {
      displayImage(result.images[0], '#canvas-i2i', '#i2i-output-image', '#i2i-output-placeholder');
      setSeedDisplay(result.seed);
      showMessage('success', `Generated. Seed: ${result.seed}`);
    }
  } catch (e) {
    showMessage('error', e.message);
  } finally {
    state.generating = false;
    setGeneratingUI(false, 'i2i');
    try {
      const status = await API.getStatus();
      updateVRAM(status.vram);
    } catch (_) {}
  }
}

// ══════════════════════════════════════════════════════════════
// INPAINT GENERATION
// ══════════════════════════════════════════════════════════════

async function runInpaint() {
  if (state.generating) return;
  dismissMessage();
  if (!state.inpaintSourceFile) {
    showMessage('error', 'Please upload a source image first.');
    return;
  }

  const maskBlob = await getMaskBlob();
  if (!maskBlob) {
    showMessage('error', 'Failed to create mask.');
    return;
  }

  const fd = new FormData();
  fd.append('source_image', state.inpaintSourceFile);
  fd.append('mask_image', maskBlob, 'mask.png');
  fd.append('positive_prompt', $('#inp-positive-prompt')?.value || '');
  fd.append('negative_prompt', $('#inp-negative-prompt')?.value || '');
  fd.append('description', $('#inp-description-prompt')?.value || '');
  fd.append('strength', $('#inp-strength')?.value || '0.7');
  fd.append('steps', $('#inp-steps-slider')?.value || '30');
  fd.append('guidance_scale', $('#inp-cfg-slider')?.value || '7.5');
  fd.append('seed', $('#inp-seed-input')?.value || '-1');
  fd.append('scheduler', $('#inp-sampler-select')?.value || 'Euler');
  fd.append('lora1_name', $('#inp-lora1-select')?.value || 'None');
  fd.append('lora1_weight', $('#inp-lora1-weight')?.value || '1.0');
  fd.append('lora2_name', $('#inp-lora2-select')?.value || 'None');
  fd.append('lora2_weight', $('#inp-lora2-weight')?.value || '1.0');

  state.generating = true;
  setGeneratingUI(true, 'inpaint');

  try {
    const result = await API.inpaint(fd);

    if (result.status === 'interrupted') {
      showMessage('info', 'Generation stopped.');
    } else if (result.images && result.images.length > 0) {
      displayImage(result.images[0], '#canvas-inpaint', '#inpaint-output-image', '#inpaint-output-placeholder');
      setSeedDisplay(result.seed);
      showMessage('success', `Generated. Seed: ${result.seed}`);
    }
  } catch (e) {
    showMessage('error', e.message);
  } finally {
    state.generating = false;
    setGeneratingUI(false, 'inpaint');
    try {
      const status = await API.getStatus();
      updateVRAM(status.vram);
    } catch (_) {}
  }
}

// ══════════════════════════════════════════════════════════════
// UI HELPERS
// ══════════════════════════════════════════════════════════════

function setGeneratingUI(generating, mode = 't2i') {
  // Mode-specific button mapping
  const btnMap = {
    t2i: { gen: '#btn-generate', stop: '#btn-stop', canvas: '#canvas-image' },
    i2i: { gen: '#btn-generate-i2i', stop: '#btn-stop-i2i', canvas: '#canvas-i2i' },
    inpaint: { gen: '#btn-generate-inpaint', stop: '#btn-stop-inpaint', canvas: '#canvas-inpaint' },
  };

  const m = btnMap[mode] || btnMap.t2i;
  const genBtn = $(m.gen);
  const stopBtn = $(m.stop);
  const canvas = $(m.canvas);

  if (genBtn) {
    genBtn.disabled = generating;
    genBtn.textContent = generating ? 'Generating...' : 'Generate';
  }
  if (stopBtn) {
    stopBtn.style.display = generating ? 'inline-block' : 'none';
  }

  // Progress bar and loading overlay within the active canvas
  if (canvas) {
    const progress = canvas.querySelector('.progress-bar-container');
    if (progress) {
      progress.classList.toggle('active', generating);
      if (generating) {
        const fill = progress.querySelector('.progress-bar-fill');
        if (fill) fill.style.width = '0%';
      }
    }
    const overlay = canvas.querySelector('.loading-overlay');
    if (overlay) overlay.classList.toggle('active', generating);
  }
}

// ── Image display ───────────────────────────────────────────

function displayImage(base64, canvasSelector = '#canvas-image', imgSelector = null, placeholderSelector = null) {
  state.currentImage = base64;

  if (imgSelector) {
    // I2I / Inpaint mode — show in dedicated output img element
    const img = $(imgSelector);
    const placeholder = $(placeholderSelector);
    if (img) {
      img.src = `data:image/png;base64,${base64}`;
      img.style.display = 'block';
      img.onclick = () => openLightbox(img.src);
    }
    if (placeholder) placeholder.style.display = 'none';
  } else {
    // T2I mode — insert into canvas-image
    const canvas = $(canvasSelector);
    if (!canvas) return;

    const placeholder = canvas.querySelector('.canvas-placeholder');
    if (placeholder) placeholder.style.display = 'none';

    let img = canvas.querySelector(':scope > img');
    if (!img) {
      img = document.createElement('img');
      img.addEventListener('click', () => openLightbox(img.src));
      canvas.insertBefore(img, canvas.firstChild);
    }
    img.style.display = '';
    img.src = `data:image/png;base64,${base64}`;
  }
}

function displayBatchGrid(images) {
  const canvas = $('#canvas-image');
  if (!canvas) return;

  // Hide placeholder and any single image
  const placeholder = canvas.querySelector('.canvas-placeholder');
  if (placeholder) placeholder.style.display = 'none';
  const singleImg = canvas.querySelector(':scope > img');
  if (singleImg) singleImg.style.display = 'none';

  // Hide old batch strip if present
  const strip = canvas.querySelector('.batch-strip');
  if (strip) strip.style.display = 'none';

  // Create or reuse grid container
  let grid = canvas.querySelector('.batch-grid');
  if (!grid) {
    grid = document.createElement('div');
    grid.className = 'batch-grid';
    canvas.insertBefore(grid, canvas.firstChild);
  }
  grid.innerHTML = '';
  grid.style.display = 'grid';

  // Choose column count based on image count
  const cols = images.length <= 2 ? 2 : images.length <= 4 ? 2 : images.length <= 6 ? 3 : 4;
  grid.style.gridTemplateColumns = `repeat(${cols}, 1fr)`;

  // Select all images by default
  state.batchIndex = 0;

  images.forEach((b64, i) => {
    const cell = document.createElement('div');
    cell.className = 'batch-cell';
    cell.dataset.index = i;

    const checkbox = document.createElement('input');
    checkbox.type = 'checkbox';
    checkbox.className = 'batch-check';
    checkbox.checked = false;
    checkbox.addEventListener('click', (e) => e.stopPropagation());
    checkbox.addEventListener('change', () => updateBatchSaveButton());

    const img = document.createElement('img');
    img.src = `data:image/png;base64,${b64}`;

    cell.addEventListener('click', () => {
      state.batchIndex = i;
      openLightbox(img.src);
    });

    cell.appendChild(checkbox);
    cell.appendChild(img);
    grid.appendChild(cell);
  });

  updateBatchSaveButton();
}

function updateBatchSaveButton() {
  const saveBtn = $('#btn-save');
  const grid = document.querySelector('.batch-grid');
  if (!grid || grid.style.display === 'none') {
    if (saveBtn) saveBtn.textContent = 'Save PNG';
    return;
  }
  const checkedCount = grid.querySelectorAll('.batch-check:checked').length;
  if (saveBtn) {
    saveBtn.textContent = checkedCount > 1 ? `Save ${checkedCount} PNGs` : 'Save PNG';
  }
}

function hideBatchGrid() {
  const canvas = $('#canvas-image');
  if (!canvas) return;
  const grid = canvas.querySelector('.batch-grid');
  if (grid) {
    grid.style.display = 'none';
    grid.innerHTML = '';
  }
  // Restore single image visibility
  const singleImg = canvas.querySelector(':scope > img');
  if (singleImg) singleImg.style.display = '';
  // Reset save button text
  updateBatchSaveButton();
}

// ── WebSocket message handler ───────────────────────────────

function handleWSMessage(data) {
  switch (data.type) {
    case 'progress':
      showMessage('info', data.message, false);
      break;
    case 'status':
      showMessage('info', data.message);
      break;
    case 'error':
      showMessage('error', data.message);
      break;
    case 'download':
      const dlProg = $('#download-progress');
      if (dlProg) dlProg.textContent = data.message;
      break;
    case 'training':
      updateTrainingLog(data.log);
      break;
    case 'step_progress': {
      const pct = Math.round(data.step / data.total * 100);
      const fill = document.querySelector('.progress-bar-container.active .progress-bar-fill');
      if (fill) fill.style.width = pct + '%';
      showMessage('info', `Step ${data.step}/${data.total}`, false);
      break;
    }
    case 'vram_update':
      updateVRAM(data.vram);
      break;
  }
}

// ══════════════════════════════════════════════════════════════
// MODEL BROWSER
// ══════════════════════════════════════════════════════════════

let _browserResults = [];
let _browserCursors = [null]; // cursor history for pagination
let _browserPage = 0;
let _selectedBrowserModel = null;

async function searchBrowser(cursor = null) {
  const grid = $('#canvas-browser');
  if (!grid) return;

  const query = $('#browser-query')?.value || '';
  const modelType = $('#browser-type')?.value || 'All';
  const baseModel = $('#browser-base')?.value || 'All';
  const sort = $('#browser-sort')?.value || 'Most Downloaded';
  const contentFilter = $('#browser-content-filter')?.value || 'Show All';
  const limit = parseInt($('#browser-per-page')?.value || '20', 10);

  grid.innerHTML = '<div style="padding:40px; text-align:center; color:var(--text-muted); grid-column:1/-1">Searching...</div>';

  try {
    const params = { query, model_type: modelType, base_model: baseModel, sort, content_filter: contentFilter, limit };
    if (cursor) params.cursor = cursor;

    const data = await API.searchCivitai(params);
    _browserResults = data.results || [];

    // Store next cursor for pagination
    if (cursor === null) {
      // Fresh search — reset pagination
      _browserCursors = [null];
      _browserPage = 0;
    }
    // Store next_cursor at current page+1 position
    if (data.next_cursor) {
      _browserCursors[_browserPage + 1] = data.next_cursor;
    }

    renderBrowserTiles(_browserResults);
    updateBrowserPagination(data.next_cursor);
  } catch (e) {
    grid.innerHTML = `<div style="padding:40px; text-align:center; color:var(--danger); grid-column:1/-1">Search failed: ${e.message}</div>`;
  }
}

function renderBrowserTiles(results) {
  const grid = $('#canvas-browser');
  if (!grid) return;
  grid.innerHTML = '';

  if (!results.length) {
    grid.innerHTML = '<div style="padding:40px; text-align:center; color:var(--text-muted); grid-column:1/-1">No results found</div>';
    return;
  }

  results.forEach(model => {
    const card = document.createElement('div');
    card.className = 'model-card';
    card.onclick = () => selectBrowserTile(card, model);

    const imgSrc = model.preview_url || '';
    const isVideo = imgSrc && (imgSrc.endsWith('.mp4') || imgSrc.includes('.mp4'));
    const typeBadge = model.type === 'LORA' ? 'LoRA' : (model.type === 'Checkpoint' ? 'Ckpt' : model.type);

    let mediaHtml;
    if (!imgSrc) {
      mediaHtml = '<div class="card-image" style="background:#333;display:flex;align-items:center;justify-content:center;color:#666;aspect-ratio:1">No Preview</div>';
    } else if (isVideo) {
      mediaHtml = `<video class="card-image" src="${imgSrc}" muted loop playsinline preload="metadata" onmouseenter="this.play()" onmouseleave="this.pause();this.currentTime=0"></video>`;
    } else {
      mediaHtml = `<img class="card-image" src="${imgSrc}" loading="lazy" onerror="this.style.display='none';this.nextElementSibling.style.display='flex';"><div class="card-image" style="display:none;background:#333;align-items:center;justify-content:center;color:#666;aspect-ratio:1">No Preview</div>`;
    }

    card.innerHTML = `
      <div class="card-image-wrap">
        ${mediaHtml}
        <span class="tile-badge">${typeBadge}</span>
      </div>
      <div class="card-info">
        <div class="card-title">${model.name}</div>
        <div class="card-meta">${model.base_model} &middot; ${model.file_size_str}</div>
      </div>
    `;
    grid.appendChild(card);
  });
}

function selectBrowserTile(card, model) {
  // Deselect others
  $$('#canvas-browser .model-card').forEach(c => c.classList.remove('selected'));
  card.classList.add('selected');

  _selectedBrowserModel = model;

  // Show detail panel
  const placeholder = $('#detail-placeholder');
  const detail = $('#model-detail');
  if (placeholder) placeholder.style.display = 'none';
  if (!detail) return;
  detail.classList.add('visible');

  // Name
  $('#detail-name').textContent = model.name;

  // Meta
  $('#detail-meta').innerHTML = `${model.type} &middot; ${model.base_model}<br>${model.version_name} &middot; ${model.file_size_str}`;

  // Description (HTML from CivitAI, sanitize by using textContent for plain text)
  const descEl = $('#detail-desc');
  if (descEl) {
    if (model.description) {
      descEl.innerHTML = model.description;
    } else {
      descEl.textContent = 'No description available.';
    }
  }

  // Tags / trigger words
  const tagsEl = $('#detail-tags');
  if (tagsEl) {
    if (model.trained_words && model.trained_words.length) {
      tagsEl.innerHTML = model.trained_words.map(w => `<span class="tag">${w}</span>`).join('');
    } else {
      tagsEl.innerHTML = '';
    }
  }

  // Trigger words (display separately for clarity)
  const triggerEl = $('#detail-trigger');
  if (triggerEl) {
    if (model.trained_words && model.trained_words.length) {
      triggerEl.innerHTML = '<strong>Trigger words:</strong> ' + model.trained_words.map(w => `<code>${w}</code>`).join(', ');
    } else {
      triggerEl.innerHTML = '';
    }
  }

  // Recommended settings
  const settingsEl = $('#detail-settings');
  if (settingsEl) {
    const rs = model.recommended_settings || {};
    const parts = [];
    if (rs.steps) parts.push(`Steps: ${rs.steps}`);
    if (rs.cfg) parts.push(`CFG: ${rs.cfg}`);
    if (rs.sampler) parts.push(`Sampler: ${rs.sampler}`);
    if (rs.clip_skip) parts.push(`Clip Skip: ${rs.clip_skip}`);
    if (parts.length) {
      settingsEl.innerHTML = '<strong>Recommended:</strong> ' + parts.join(' &middot; ');
    } else {
      settingsEl.innerHTML = '';
    }
  }

  // Links
  const linksEl = $('#detail-links');
  if (linksEl) {
    const hfQuery = encodeURIComponent(model.name);
    linksEl.innerHTML =
      `<a href="${model.civitai_url}" target="_blank" rel="noopener">View on CivitAI</a>` +
      `<a href="https://huggingface.co/models?search=${hfQuery}" target="_blank" rel="noopener" style="color:#f59e0b">Search on HuggingFace</a>`;
  }

  // Clear download progress
  const dlProg = $('#download-progress');
  if (dlProg) dlProg.textContent = '';
}

async function downloadBrowserModel() {
  if (!_selectedBrowserModel) return;
  const model = _selectedBrowserModel;

  const dlProg = $('#download-progress');
  const dlBtn = $('#btn-download');
  if (dlBtn) dlBtn.disabled = true;
  if (dlProg) dlProg.textContent = 'Starting download...';

  try {
    const metadata = {
      trained_words: model.trained_words || [],
      recommended_settings: model.recommended_settings || {},
      civitai_url: model.civitai_url || '',
    };

    await API.downloadCivitai({
      download_url: model.download_url,
      model_type: model.type === 'LORA' ? 'LORA' : 'Checkpoint',
      base_model: model.base_model,
      filename: model.filename,
      metadata: model.type === 'LORA' ? metadata : null,
    });

    if (dlProg) dlProg.textContent = 'Download complete!';
    showMessage('success', `Downloaded ${model.filename}`);
  } catch (e) {
    if (dlProg) dlProg.textContent = 'Download failed.';
    showMessage('error', `Download failed: ${e.message}`);
  } finally {
    if (dlBtn) dlBtn.disabled = false;
  }
}

function updateBrowserPagination(nextCursor) {
  const pag = $('#browser-pagination');
  if (!pag) return;
  pag.style.display = 'flex';

  const prevBtn = $('#btn-browser-prev');
  const nextBtn = $('#btn-browser-next');
  const indicator = $('#browser-page-indicator');

  if (prevBtn) prevBtn.disabled = (_browserPage === 0);
  if (nextBtn) nextBtn.disabled = !nextCursor;
  if (indicator) indicator.textContent = `Page ${_browserPage + 1}`;
}

async function browserNextPage() {
  const nextCursor = _browserCursors[_browserPage + 1];
  if (!nextCursor) return;
  _browserPage++;
  await searchBrowser(nextCursor);
}

async function browserPrevPage() {
  if (_browserPage <= 0) return;
  _browserPage--;
  const cursor = _browserCursors[_browserPage] || null;
  await searchBrowser(cursor);
}

// ── Preview Files ───────────────────────────────────────────

async function loadPreviewFiles() {
  const grid = $('#canvas-preview');
  if (!grid) return;

  const filterType = $('#preview-filter-type')?.value || 'All';
  const sortOrder = $('#preview-sort')?.value || 'Newest First';

  try {
    const data = await API.getOutputs(filterType, sortOrder);
    grid.innerHTML = '';

    if (!data.files || data.files.length === 0) {
      grid.innerHTML = '<div style="padding:40px; text-align:center; color:var(--text-muted); grid-column:1/-1">No output files yet</div>';
      return;
    }

    data.files.forEach(file => {
      const card = document.createElement('div');
      card.className = 'preview-card';
      card.onclick = () => selectPreviewCard(card, file);

      const isVideo = file.name.endsWith('.mp4');
      const thumbContent = isVideo
        ? `<img src="${API.getOutputThumbUrl(file.name)}" loading="lazy" onerror="this.outerHTML='<div style=\\'font-size:24px;color:#444\\'>&#127916;</div>'">`
        : `<img src="${API.getOutputUrl(file.name)}" loading="lazy">`;

      card.innerHTML = `
        <input type="checkbox" class="card-check" onclick="event.stopPropagation(); updateBatchDeleteCount()">
        <div class="card-thumb">${thumbContent}</div>
        <div class="card-name">${file.name}</div>
      `;
      grid.appendChild(card);
    });
  } catch (e) {
    console.error('Failed to load preview files:', e);
  }
}

function selectPreviewCard(card, file) {
  // Deselect other cards
  $$('#canvas-preview .preview-card').forEach(c => c.classList.remove('selected'));
  card.classList.add('selected');

  // Show detail in sidebar
  const placeholder = $('#file-detail-placeholder');
  const detail = $('#file-detail');
  if (placeholder) placeholder.style.display = 'none';
  if (!detail) return;

  detail.classList.add('visible');

  const isVideo = file.name.endsWith('.mp4');

  // Preview image or video
  const preview = detail.querySelector('.fd-preview');
  if (preview) {
    if (isVideo) {
      preview.innerHTML = `<video src="${API.getOutputUrl(file.name)}" controls loop style="width:100%;border-radius:6px"></video>`;
    } else {
      preview.innerHTML = `<img src="${API.getOutputUrl(file.name)}">`;
    }
  }

  // Name
  const nameEl = detail.querySelector('.fd-name');
  if (nameEl) nameEl.textContent = file.name;

  // Meta
  const metaEl = detail.querySelector('.fd-meta');
  if (metaEl) {
    const sizeKB = (file.size / 1024).toFixed(1);
    const sizeMB = (file.size / (1024 * 1024)).toFixed(1);
    const sizeStr = file.size > 1024 * 1024 ? `${sizeMB} MB` : `${sizeKB} KB`;
    const date = new Date(file.modified * 1000).toLocaleString();
    metaEl.innerHTML = `Size: ${sizeStr}<br>Modified: ${date}`;
  }

  // Params (from metadata)
  const paramsEl = detail.querySelector('.fd-params');
  if (paramsEl) {
    if (file.metadata) {
      const m = file.metadata;
      paramsEl.innerHTML = [
        m.positive_prompt ? `<div><span class="param-label">Prompt:</span> ${m.positive_prompt}</div>` : '',
        m.negative_prompt ? `<div><span class="param-label">Negative:</span> ${m.negative_prompt}</div>` : '',
        `<div><span class="param-label">Steps:</span> ${m.steps || '?'} &middot; <span class="param-label">CFG:</span> ${m.guidance_scale || '?'}</div>`,
        `<div><span class="param-label">Seed:</span> ${m.seed || '?'} &middot; <span class="param-label">Sampler:</span> ${m.sampler || '?'}</div>`,
        m.model ? `<div><span class="param-label">Model:</span> ${m.model}</div>` : '',
      ].join('');
      paramsEl.style.display = 'block';
    } else {
      paramsEl.style.display = 'none';
    }
  }

  // Wire Open button
  const openBtn = $('#btn-open-file');
  if (openBtn) {
    openBtn.onclick = () => window.open(API.getOutputUrl(file.name), '_blank');
  }

  // Wire delete button
  const deleteBtn = detail.querySelector('.btn-danger');
  if (deleteBtn) {
    deleteBtn.onclick = async () => {
      if (!confirm(`Delete ${file.name}?`)) return;
      try {
        await API.deleteOutput(file.name);
        showMessage('success', `Deleted ${file.name}`);
        loadPreviewFiles();
        detail.classList.remove('visible');
        if (placeholder) placeholder.style.display = 'block';
      } catch (e) {
        showMessage('error', e.message);
      }
    };
  }
}

function updateBatchDeleteCount() {
  const btn = $('#btn-batch-delete');
  if (!btn) return;
  const count = $$('#canvas-preview .card-check:checked').length;
  btn.textContent = `Delete (${count})`;
  btn.disabled = count === 0;
}

// ══════════════════════════════════════════════════════════════
// VIDEO GENERATION
// ══════════════════════════════════════════════════════════════

async function loadVideoModels() {
  try {
    const data = await API.getVideoArchitectures();
    if (data.models) {
      populateSelect('#video-model-select', data.models, '(none)');
    }
    // Also try to load models for current arch
    const models = await API.getVideoModels();
    populateSelect('#video-model-select', models.models, '(none)');
    if (models.active) {
      $('#video-model-select').value = models.active;
    }
  } catch (e) {
    // Not critical - video tab may not be visible yet
  }
}

function applyVideoDefaults(defaults) {
  if (!defaults) return;
  const steps = $('#video-steps');
  if (steps && defaults.steps) { steps.value = defaults.steps; steps.nextElementSibling.textContent = defaults.steps; }
  const cfg = $('#video-cfg');
  if (cfg && defaults.guidance_scale !== undefined) { cfg.value = defaults.guidance_scale; cfg.nextElementSibling.textContent = defaults.guidance_scale; }
  const fps = $('#video-fps');
  if (fps && defaults.fps) { fps.value = defaults.fps; fps.nextElementSibling.textContent = defaults.fps; }
  if (defaults.schedulers) {
    populateSelect('#video-scheduler', defaults.schedulers);
  }
  if (defaults.scheduler) {
    const sel = $('#video-scheduler');
    if (sel) sel.value = defaults.scheduler;
  }
}

async function updateVideoVramEstimate() {
  try {
    const duration = parseFloat($('#video-duration')?.value || 2);
    const fps = parseInt($('#video-fps')?.value || 24);
    const data = await API.getVideoVramEstimate(duration, fps);
    const el = $('#video-vram-est');
    if (el) el.textContent = data.estimate;
  } catch (e) {
    // Ignore
  }
}

async function runVideoGeneration() {
  if (state.videoGenerating) return;
  dismissMessage();

  const params = {
    positive_prompt: $('#video-positive-prompt')?.value || '',
    negative_prompt: $('#video-negative-prompt')?.value || '',
    duration: parseFloat($('#video-duration')?.value || 2),
    fps: parseInt($('#video-fps')?.value || 24),
    steps: parseInt($('#video-steps')?.value || 25),
    guidance_scale: parseFloat($('#video-cfg')?.value || 9.0),
    seed: parseInt($('#video-seed')?.value || -1),
    scheduler: $('#video-scheduler')?.value || 'UniPC',
  };

  state.videoGenerating = true;
  setVideoGeneratingUI(true);

  try {
    const result = await API.generateVideo(params);

    if (result.status === 'interrupted') {
      showMessage('info', 'Video generation stopped.');
    } else if (result.video_url) {
      displayVideo(result.video_url);
      showMessage('success', `Video generated. Seed: ${result.seed} | ${result.num_frames} frames`);
    }
  } catch (e) {
    showMessage('error', e.message);
  } finally {
    state.videoGenerating = false;
    setVideoGeneratingUI(false);
    try {
      const status = await API.getStatus();
      updateVRAM(status.vram);
    } catch (_) {}
  }
}

function displayVideo(url) {
  const player = $('#video-player');
  const placeholder = $('#video-placeholder');
  if (player) {
    player.src = url;
    player.style.display = 'block';
    player.load();
  }
  if (placeholder) placeholder.style.display = 'none';
}

function setVideoGeneratingUI(generating) {
  const genBtn = $('#btn-generate-video');
  const stopBtn = $('#btn-stop-video');
  if (genBtn) {
    genBtn.disabled = generating;
    genBtn.textContent = generating ? 'Generating...' : 'Generate Video';
  }
  if (stopBtn) stopBtn.style.display = generating ? 'inline-block' : 'none';
}

// ══════════════════════════════════════════════════════════════
// ANIMATEDIFF
// ══════════════════════════════════════════════════════════════

async function loadAnimateModelLists() {
  try {
    const data = await API.getAnimateModels();
    populateSelect('#anim-base-model', data.base_models, '(select)');
    populateSelect('#anim-motion-adapter', data.motion_adapters, '(select)');
    populateSelect('#anim-sparsectrl', data.sparsectrls, '(select)');
  } catch (e) {
    // Not critical
  }
}

async function loadAnimateDiffModels() {
  const base = $('#anim-base-model')?.value;
  const motion = $('#anim-motion-adapter')?.value;
  const sparse = $('#anim-sparsectrl')?.value;

  if (!base || base === '(select)') { showMessage('error', 'Select a base model.'); return; }
  if (!motion || motion === '(select)') { showMessage('error', 'Select a motion adapter.'); return; }
  if (!sparse || sparse === '(select)') { showMessage('error', 'Select a SparseCtrl model.'); return; }

  try {
    showMessage('info', 'Loading AnimateDiff models...');
    const result = await API.loadAnimateModels(base, motion, sparse);
    const statusEl = $('#anim-model-status');
    if (statusEl) statusEl.textContent = result.status;
    showMessage('success', result.status);
    try { const s = await API.getStatus(); updateVRAM(s.vram); } catch (_) {}
  } catch (e) {
    showMessage('error', e.message);
  }
}

function initAnimSourceUpload() {
  const dropzone = $('#anim-source-dropzone');
  if (!dropzone) return;

  // Create hidden file input
  const fileInput = document.createElement('input');
  fileInput.type = 'file';
  fileInput.accept = 'image/*';
  fileInput.style.display = 'none';
  dropzone.appendChild(fileInput);

  dropzone.addEventListener('click', (e) => {
    if (e.target.id === 'anim-clear-source') return;
    fileInput.click();
  });

  fileInput.addEventListener('change', () => {
    if (fileInput.files.length > 0) handleAnimFile(fileInput.files[0]);
  });

  dropzone.addEventListener('dragover', (e) => { e.preventDefault(); dropzone.classList.add('drag-over'); });
  dropzone.addEventListener('dragleave', () => dropzone.classList.remove('drag-over'));
  dropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropzone.classList.remove('drag-over');
    if (e.dataTransfer.files.length > 0) handleAnimFile(e.dataTransfer.files[0]);
  });

  const clearBtn = $('#anim-clear-source');
  if (clearBtn) {
    clearBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      clearAnimSource();
    });
  }
}

function handleAnimFile(file) {
  if (!file.type.startsWith('image/')) {
    showMessage('error', 'Please upload an image file.');
    return;
  }
  state.animSourceFile = file;
  const preview = $('#anim-source-preview');
  const icon = $('#anim-source-icon');
  const hint = $('#anim-source-hint');
  const clearBtn = $('#anim-clear-source');

  if (preview) {
    preview.src = URL.createObjectURL(file);
    preview.style.display = 'block';
  }
  if (icon) icon.style.display = 'none';
  if (hint) hint.style.display = 'none';
  if (clearBtn) clearBtn.style.display = 'block';
}

function clearAnimSource() {
  state.animSourceFile = null;
  const preview = $('#anim-source-preview');
  const icon = $('#anim-source-icon');
  const hint = $('#anim-source-hint');
  const clearBtn = $('#anim-clear-source');

  if (preview) { preview.style.display = 'none'; preview.src = ''; }
  if (icon) icon.style.display = 'block';
  if (hint) hint.style.display = 'block';
  if (clearBtn) clearBtn.style.display = 'none';
}

async function updateAnimVramEstimate() {
  try {
    const frames = parseInt($('#anim-frames')?.value || 16);
    const fps = parseInt($('#anim-fps')?.value || 12);
    const data = await API.getAnimateVramEstimate(frames, fps);
    const el = $('#anim-vram-est');
    if (el) el.textContent = data.estimate;
  } catch (e) {
    // Ignore
  }
}

async function runAnimation() {
  if (state.animGenerating) return;
  dismissMessage();
  if (!state.animSourceFile) {
    showMessage('error', 'Please upload a source image first.');
    return;
  }

  const fd = new FormData();
  fd.append('source_image', state.animSourceFile);
  fd.append('positive_prompt', $('#anim-positive-prompt')?.value || '');
  fd.append('negative_prompt', $('#anim-negative-prompt')?.value || '');
  fd.append('num_frames', $('#anim-frames')?.value || '16');
  fd.append('fps', $('#anim-fps')?.value || '12');
  fd.append('steps', $('#anim-steps')?.value || '25');
  fd.append('guidance_scale', $('#anim-cfg')?.value || '7.5');
  fd.append('conditioning_scale', $('#anim-cond')?.value || '1.0');
  fd.append('seed', $('#anim-seed')?.value || '-1');
  fd.append('scheduler', $('#anim-scheduler')?.value || 'Euler');

  state.animGenerating = true;
  setAnimGeneratingUI(true);

  try {
    const result = await API.generateAnimation(fd);

    if (result.status === 'interrupted') {
      showMessage('info', 'Animation stopped.');
    } else if (result.video_url) {
      displayAnimVideo(result.video_url);
      showMessage('success', `Animation generated. Seed: ${result.seed} | ${result.num_frames} frames`);
    }
  } catch (e) {
    showMessage('error', e.message);
  } finally {
    state.animGenerating = false;
    setAnimGeneratingUI(false);
    try {
      const status = await API.getStatus();
      updateVRAM(status.vram);
    } catch (_) {}
  }
}

function displayAnimVideo(url) {
  const player = $('#anim-player');
  const icon = $('#anim-output-icon');
  const hint = $('#anim-output-hint');
  if (player) {
    player.src = url;
    player.style.display = 'block';
    player.load();
  }
  if (icon) icon.style.display = 'none';
  if (hint) hint.style.display = 'none';
}

function setAnimGeneratingUI(generating) {
  const genBtn = $('#btn-animate');
  const stopBtn = $('#btn-stop-anim');
  if (genBtn) {
    genBtn.disabled = generating;
    genBtn.textContent = generating ? 'Animating...' : 'Animate';
  }
  if (stopBtn) stopBtn.style.display = generating ? 'inline-block' : 'none';
}

// ══════════════════════════════════════════════════════════════
// LORA TRAINING
// ══════════════════════════════════════════════════════════════

let _trainingInProgress = false;

async function startTraining() {
  if (_trainingInProgress) return;

  const imageDir = $('#train-folder')?.value?.trim();
  const outputName = $('#train-name')?.value?.trim();
  const steps = parseInt($('#train-steps')?.value || '500');
  const lrExponent = parseFloat($('#train-lr')?.value || '-4');
  const learningRate = Math.pow(10, lrExponent);
  const rank = parseInt($('#train-rank')?.value || '4');

  if (!imageDir) { showMessage('error', 'Please enter an image folder path.'); return; }
  if (!outputName) { showMessage('error', 'Please enter an output name.'); return; }

  _trainingInProgress = true;
  setTrainingUI(true);

  // Clear log
  const log = $('#training-log');
  if (log) log.innerHTML = '';
  updateTrainingProgress(0, 'Starting...');

  try {
    const result = await API.startTraining({
      image_dir: imageDir,
      output_name: outputName,
      steps, learning_rate: learningRate, rank,
    });
    if (result.path) {
      showMessage('success', `LoRA saved: ${result.path}`);
    }
  } catch (e) {
    showMessage('error', e.message);
  } finally {
    _trainingInProgress = false;
    setTrainingUI(false);
  }
}

async function stopTraining() {
  try {
    await API.stopTraining();
    showMessage('info', 'Stopping training...');
  } catch (e) {
    showMessage('error', e.message);
  }
}

function setTrainingUI(training) {
  const startBtn = $('#btn-train');
  const stopBtn = $('#btn-stop-train');
  if (startBtn) {
    startBtn.disabled = training;
    startBtn.textContent = training ? 'Training...' : 'Start Training';
  }
  if (stopBtn) stopBtn.disabled = !training;
}

function updateTrainingLog(logText) {
  const log = $('#training-log');
  if (!log) return;

  const lines = logText.split('\n');
  log.innerHTML = '';
  lines.forEach(line => {
    const div = document.createElement('div');
    // Highlight step/loss with CSS classes
    const stepMatch = line.match(/Step (\d+)\/(\d+)/);
    if (stepMatch) {
      const current = parseInt(stepMatch[1]);
      const total = parseInt(stepMatch[2]);
      div.innerHTML = line
        .replace(/(Step \d+\/\d+)/, '<span class="log-step">$1</span>')
        .replace(/(loss: [\d.]+)/, '<span class="log-loss">$1</span>');
      updateTrainingProgress(current / total * 100, `Step ${current}/${total}`);
    } else {
      div.textContent = line;
    }
    log.appendChild(div);
  });
  log.scrollTop = log.scrollHeight;
}

function updateTrainingProgress(pct, text) {
  const fill = document.querySelector('.tp-fill');
  const tpText = document.querySelector('.tp-text');
  if (fill) fill.style.width = pct + '%';
  if (tpText) tpText.textContent = text;
}


// ══════════════════════════════════════════════════════════════
// PROMPT PROFILES
// ══════════════════════════════════════════════════════════════

async function loadProfiles(autoLoadDefault = false) {
  try {
    const data = await API.getProfiles();
    const sel = $('#profile-select');
    if (!sel) return;
    const current = sel.value;
    sel.innerHTML = '<option value="">Select profile...</option>';
    const profiles = data.profiles || [];
    profiles.forEach(name => {
      const opt = document.createElement('option');
      opt.value = name;
      opt.textContent = name;
      sel.appendChild(opt);
    });
    if (current) sel.value = current;

    // Auto-load default profile into all prompt fields on startup
    if (autoLoadDefault && profiles.includes('default')) {
      const defaults = await API.loadProfile('default');
      const pos = defaults.positive || '';
      const neg = defaults.negative || '';
      // Fill all prompt textareas (t2i, i2i, inpaint)
      ['#positive-prompt', '#i2i-positive-prompt', '#inp-positive-prompt'].forEach(id => {
        const el = $(id);
        if (el && !el.value) el.value = pos;
      });
      ['#negative-prompt', '#i2i-negative-prompt', '#inp-negative-prompt'].forEach(id => {
        const el = $(id);
        if (el && !el.value) el.value = neg;
      });
    }
  } catch (e) {
    // Not critical
  }
}

function _getActivePromptFields() {
  // Determine which prompt textareas to use based on current mode
  const mode = state.mode;
  const sub = state.subMode || mode;
  if (sub === 'i2i') return { pos: '#i2i-positive-prompt', neg: '#i2i-negative-prompt' };
  if (sub === 'inpaint') return { pos: '#inp-positive-prompt', neg: '#inp-negative-prompt' };
  if (sub === 't2v') return { pos: '#video-positive-prompt', neg: '#video-negative-prompt' };
  if (sub === 'animate') return { pos: '#anim-positive-prompt', neg: '#anim-negative-prompt' };
  // Default: t2i
  return { pos: '#positive-prompt', neg: '#negative-prompt' };
}

async function loadSelectedProfile() {
  const name = $('#profile-select')?.value;
  if (!name) { showMessage('error', 'No profile selected.'); return; }
  try {
    const data = await API.loadProfile(name);
    const fields = _getActivePromptFields();
    const posEl = $(fields.pos);
    const negEl = $(fields.neg);
    if (posEl) posEl.value = data.positive || '';
    if (negEl) negEl.value = data.negative || '';
    showMessage('success', `Profile "${name}" loaded.`);
  } catch (e) {
    showMessage('error', e.message);
  }
}

function showSaveProfileInput() {
  const group = $('#profile-save-group');
  if (!group) return;
  const isVisible = group.style.display !== 'none';
  group.style.display = isVisible ? 'none' : 'block';
  if (!isVisible) {
    const nameInput = $('#profile-save-name');
    const selected = $('#profile-select')?.value;
    if (nameInput) {
      nameInput.value = selected || '';
      nameInput.focus();
    }
  }
}

async function confirmSaveProfile() {
  const name = $('#profile-save-name')?.value?.trim();
  if (!name) { showMessage('error', 'Please enter a profile name.'); return; }

  const fields = _getActivePromptFields();
  const positive = $(fields.pos)?.value || '';
  const negative = $(fields.neg)?.value || '';

  try {
    const data = await API.saveProfile(name, { positive, negative });
    // Update dropdown
    const sel = $('#profile-select');
    if (sel && data.profiles) {
      sel.innerHTML = '<option value="">Select profile...</option>';
      data.profiles.forEach(n => {
        const opt = document.createElement('option');
        opt.value = n;
        opt.textContent = n;
        sel.appendChild(opt);
      });
      sel.value = name;
    }
    $('#profile-save-group').style.display = 'none';
    showMessage('success', `Profile "${name}" saved.`);
  } catch (e) {
    showMessage('error', e.message);
  }
}

async function deleteSelectedProfile() {
  const name = $('#profile-select')?.value;
  if (!name) { showMessage('error', 'No profile selected.'); return; }
  if (!confirm(`Delete profile "${name}"?`)) return;

  try {
    const data = await API.deleteProfile(name);
    const sel = $('#profile-select');
    if (sel && data.profiles) {
      sel.innerHTML = '<option value="">Select profile...</option>';
      data.profiles.forEach(n => {
        const opt = document.createElement('option');
        opt.value = n;
        opt.textContent = n;
        sel.appendChild(opt);
      });
    }
    showMessage('success', name === 'default' ? 'Default profile cleared.' : `Profile "${name}" deleted.`);
  } catch (e) {
    showMessage('error', e.message);
  }
}


// ── Lightbox (full-size image viewer) ───────────────────────

function openLightbox(src) {
  const lb = $('#lightbox');
  if (!lb) return;
  const img = lb.querySelector('img');
  img.src = src;
  lb.classList.add('active');
}

function closeLightbox() {
  const lb = $('#lightbox');
  if (lb) lb.classList.remove('active');
}

// Wire lightbox close events
document.addEventListener('DOMContentLoaded', () => {
  const lb = $('#lightbox');
  if (!lb) return;

  // Click backdrop or close button
  lb.addEventListener('click', (e) => {
    if (e.target === lb || e.target.classList.contains('lightbox-close')) {
      closeLightbox();
    }
  });

  // Global keyboard shortcuts
  document.addEventListener('keydown', (e) => {
    // Escape — close lightbox or stop generation
    if (e.key === 'Escape') {
      if (lb.classList.contains('active')) {
        closeLightbox();
        return;
      }
      // Stop active generation based on current mode
      const sub = state.subMode || state.mode;
      const stopMap = {
        t2i: '#btn-stop', i2i: '#btn-stop-i2i', inpaint: '#btn-stop-inpaint',
        t2v: '#btn-stop-video', animate: '#btn-stop-anim', train: '#btn-stop-train',
      };
      const stopBtn = $(stopMap[sub]);
      if (stopBtn && stopBtn.style.display !== 'none') stopBtn.click();
      return;
    }

    // Ctrl+Enter — trigger generate
    if (e.ctrlKey && e.key === 'Enter') {
      e.preventDefault();
      const sub = state.subMode || state.mode;
      const btnMap = {
        t2i: '#btn-generate', i2i: '#btn-generate-i2i', inpaint: '#btn-generate-inpaint',
        t2v: '#btn-generate-video', animate: '#btn-animate', train: '#btn-train',
      };
      const btn = $(btnMap[sub]);
      if (btn && !btn.disabled) btn.click();
      return;
    }

    // Ctrl+S — save current output (prevent browser save dialog)
    if (e.ctrlKey && e.key === 's') {
      e.preventDefault();
      const sub = state.subMode || state.mode;
      const saveMap = {
        t2i: '#btn-save', i2i: '#btn-save-i2i', inpaint: '#btn-save-inpaint',
        t2v: '#btn-save-video', animate: '#btn-save-anim',
      };
      const btn = $(saveMap[sub]);
      if (btn) btn.click();
      return;
    }
  });
});
