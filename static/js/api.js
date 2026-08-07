/**
 * ImaGen API Client — fetch wrappers + WebSocket for progress.
 */

const API = {
  /** Base URL (same origin) */
  base: '',

  /** Active WebSocket connection */
  ws: null,

  /** Registered progress/status handlers */
  _handlers: [],

  // ── WebSocket ─────────────────────────────────────────────

  connectWebSocket() {
    const proto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url = `${proto}//${location.host}/ws/progress`;
    this.ws = new WebSocket(url);

    this.ws.onmessage = (evt) => {
      try {
        const data = JSON.parse(evt.data);
        this._handlers.forEach(fn => fn(data));
      } catch (e) {
        console.error('WebSocket parse error:', e);
      }
    };

    this.ws.onclose = () => {
      // Auto-reconnect after 2s
      setTimeout(() => this.connectWebSocket(), 2000);
    };

    this.ws.onerror = () => {
      this.ws.close();
    };
  },

  /** Register a handler for WebSocket messages */
  onProgress(fn) {
    this._handlers.push(fn);
  },

  // ── Generic fetch helpers ─────────────────────────────────

  async get(path) {
    const res = await fetch(`${this.base}${path}`);
    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: res.statusText }));
      throw new Error(err.error || res.statusText);
    }
    return res.json();
  },

  async post(path, body = {}) {
    const res = await fetch(`${this.base}${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: res.statusText }));
      throw new Error(err.error || res.statusText);
    }
    return res.json();
  },

  async postForm(path, formData) {
    const res = await fetch(`${this.base}${path}`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: res.statusText }));
      throw new Error(err.error || res.statusText);
    }
    return res.json();
  },

  async del(path) {
    const res = await fetch(`${this.base}${path}`, { method: 'DELETE' });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ error: res.statusText }));
      throw new Error(err.error || res.statusText);
    }
    return res.json();
  },

  // ── Status ────────────────────────────────────────────────

  getStatus() {
    return this.get('/api/status');
  },

  // ── Architecture ──────────────────────────────────────────

  getArchitectures() {
    return this.get('/api/architectures');
  },

  switchArchitecture(arch) {
    return this.post('/api/architecture', { architecture: arch });
  },

  // ── Models ────────────────────────────────────────────────

  getModels() {
    return this.get('/api/models');
  },

  loadModel(model) {
    return this.post('/api/model', { model });
  },

  // ── LoRAs ─────────────────────────────────────────────────

  getLoras() {
    return this.get('/api/loras');
  },

  getLoraTriggers(name) {
    return this.get(`/api/lora-triggers?name=${encodeURIComponent(name)}`);
  },

  // ── VAEs ──────────────────────────────────────────────────

  getVaes() {
    return this.get('/api/vaes');
  },

  loadVae(vae) {
    return this.post('/api/vae', { vae });
  },

  // ── Upscalers ─────────────────────────────────────────────

  getUpscalers() {
    return this.get('/api/upscalers');
  },

  // ── Schedulers ────────────────────────────────────────────

  getSchedulers() {
    return this.get('/api/schedulers');
  },

  // ── Generation ────────────────────────────────────────────

  generate(params) {
    return this.post('/api/generate', params);
  },

  img2img(formData) {
    return this.postForm('/api/img2img', formData);
  },

  inpaint(formData) {
    return this.postForm('/api/inpaint', formData);
  },

  interrupt() {
    return this.post('/api/interrupt');
  },

  // ── Save ──────────────────────────────────────────────────

  saveImage(mode, saveHistory, index = null) {
    const body = { mode, save_history: saveHistory };
    if (index !== null) body.index = index;
    return this.post('/api/save', body);
  },

  saveAll(saveHistory) {
    return this.post('/api/save-all', { save_history: saveHistory });
  },

  // ── Outputs ───────────────────────────────────────────────

  getOutputs(filterType = 'All', sortOrder = 'Newest First') {
    const params = new URLSearchParams({ filter_type: filterType, sort_order: sortOrder });
    return this.get(`/api/outputs?${params}`);
  },

  getOutputUrl(filename) {
    return `/api/outputs/${encodeURIComponent(filename)}`;
  },

  getOutputThumbUrl(filename) {
    return `/api/outputs/thumb/${encodeURIComponent(filename)}`;
  },

  deleteOutput(filename) {
    return this.del(`/api/outputs/${encodeURIComponent(filename)}`);
  },

  deleteOutputsBatch(filenames) {
    return this.post('/api/outputs/delete-batch', { filenames });
  },

  // ── Video ──────────────────────────────────────────────────

  getVideoArchitectures() {
    return this.get('/api/video/architectures');
  },

  switchVideoArchitecture(arch) {
    return this.post('/api/video/architecture', { architecture: arch });
  },

  getVideoModels() {
    return this.get('/api/video/models');
  },

  loadVideoModel(model) {
    return this.post('/api/video/model', { model });
  },

  getVideoLoras() {
    return this.get('/api/video/loras');
  },

  getVideoVramEstimate(duration, fps) {
    return this.get(`/api/video/vram-estimate?duration=${duration}&fps=${fps}`);
  },

  generateVideo(params) {
    return this.post('/api/video/generate', params);
  },

  getVideoPreviewUrl() {
    return `/api/video/preview?t=${Date.now()}`;
  },

  interruptVideo() {
    return this.post('/api/video/interrupt');
  },

  saveVideo() {
    return this.post('/api/video/save');
  },

  // ── AnimateDiff ────────────────────────────────────────────

  getAnimateModels() {
    return this.get('/api/animate/models');
  },

  loadAnimateModels(base_model, motion_adapter, sparsectrl) {
    return this.post('/api/animate/load', { base_model, motion_adapter, sparsectrl });
  },

  getAnimateVramEstimate(num_frames, fps) {
    return this.get(`/api/animate/vram-estimate?num_frames=${num_frames}&fps=${fps}`);
  },

  generateAnimation(formData) {
    return this.postForm('/api/animate/generate', formData);
  },

  getAnimatePreviewUrl() {
    return `/api/animate/preview?t=${Date.now()}`;
  },

  interruptAnimation() {
    return this.post('/api/animate/interrupt');
  },

  saveAnimation() {
    return this.post('/api/animate/save');
  },

  // ── CivitAI Browser ──────────────────────────────────────

  getCivitaiEnabled() {
    return this.get('/api/civitai/enabled');
  },

  setCivitaiEnabled(enabled) {
    return this.post('/api/civitai/enabled', { enabled });
  },

  searchCivitai(params) {
    const qs = new URLSearchParams(params);
    return this.get(`/api/civitai/search?${qs}`);
  },

  downloadCivitai(body) {
    return this.post('/api/civitai/download', body);
  },

  getCivitaiKey() {
    return this.get('/api/civitai/apikey');
  },

  saveCivitaiKey(key) {
    return this.post('/api/civitai/apikey', { key });
  },

  // ── LoRA Training ────────────────────────────────────────

  startTraining(body) {
    return this.post('/api/train/start', body);
  },

  stopTraining() {
    return this.post('/api/train/stop');
  },

  // ── Profiles ─────────────────────────────────────────────

  getProfiles() {
    return this.get('/api/profiles');
  },

  loadProfile(name) {
    return this.get(`/api/profiles/${encodeURIComponent(name)}`);
  },

  saveProfile(name, body) {
    return this.post(`/api/profiles/${encodeURIComponent(name)}`, body);
  },

  deleteProfile(name) {
    return this.del(`/api/profiles/${encodeURIComponent(name)}`);
  },

  // ── Shutdown ──────────────────────────────────────────────

  shutdown() {
    return this.post('/api/shutdown');
  },
};
