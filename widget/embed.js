(function(window) {
  'use strict';

  const DEFAULT_API_BASE = window.location.origin;

  const STAGE_LABELS = {
    "queued": "Queued in GPU worker cluster...",
    "parsing": "Stage 1: Analyzing body landmarks & segmenting clothing...",
    "pre_inpaint": "Stage 2: Removing kurta flaps & reconstructing lower wear...",
    "diffusion": "Stage 3: Draping target garment with diffusion flow...",
    "blending": "Stage 4: Polishing details & restoring original face...",
    "completed": "Complete!"
  };

  class VirtualTryOnClient {
    constructor() {
      this.apiBase = DEFAULT_API_BASE;
      this.storeId = "demo_store";
      this.activeGarment = null;
      this.selectedUserImageB64 = null;
      this.isProcessing = false;
      this.eventSource = null;

      this.initModal();
    }

    init(config = {}) {
      if (config.apiBase) this.apiBase = config.apiBase.replace(/\/$/, "");
      if (config.storeId) this.storeId = config.storeId;
    }

    initModal() {
      if (document.getElementById('vton-modal-root')) return;

      const modalHtml = `
        <div id="vton-modal-root" class="vton-modal-backdrop">
          <div class="vton-modal">
            <div class="vton-header">
              <h3>
                Virtual Try-On 
                <span class="vton-badge">AI Powered</span>
              </h3>
              <button class="vton-close-btn" id="vton-close-btn">&times;</button>
            </div>
            
            <div class="vton-body">
              <!-- Left Column: Inputs -->
              <div class="vton-panel">
                <label style="font-weight: 600; font-size: 0.9rem;">1. Your Photo</label>
                <div class="vton-preview-box" id="vton-dropzone">
                  <div class="vton-upload-prompt" id="vton-upload-prompt">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                      <rect width="18" height="18" x="3" y="3" rx="2" ry="2"/>
                      <circle cx="9" cy="9" r="2"/>
                      <path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21"/>
                    </svg>
                    <p style="margin: 0; font-weight: 600;">Click to upload your full-body photo</p>
                    <p style="margin: 4px 0 0 0; font-size: 0.8rem;">Supports JPG, PNG, WebP (Kurta, shirts, etc.)</p>
                  </div>
                  <img id="vton-user-preview" style="display:none;" />
                  <input type="file" id="vton-file-input" accept="image/*" style="display:none;" />
                </div>

                <div style="display: flex; gap: 8px;">
                  <button class="vton-btn" id="vton-sample-kurta-btn" style="background: #4b5563; font-size: 0.85rem; padding: 8px 12px;">
                    Use Sample Photo (Wearing Kurta)
                  </button>
                </div>
              </div>

              <!-- Right Column: Product & Result -->
              <div class="vton-panel">
                <label style="font-weight: 600; font-size: 0.9rem;">2. Target Garment & Try-On</label>
                <div class="vton-preview-box" style="min-height: 220px; cursor: default;">
                  <img id="vton-garment-preview" style="object-fit: contain; padding: 12px;" />
                </div>

                <!-- Progress Bar -->
                <div class="vton-progress-container" id="vton-progress-box">
                  <div class="vton-progress-stage">
                    <span id="vton-stage-text">Preparing generation...</span>
                    <span id="vton-stage-pct">0%</span>
                  </div>
                  <div class="vton-progress-bar-bg">
                    <div class="vton-progress-bar-fill" id="vton-progress-fill"></div>
                  </div>
                </div>

                <!-- Action / Comparison -->
                <button class="vton-btn" id="vton-submit-btn" disabled>
                  Generate Virtual Try-On
                </button>
              </div>
            </div>

            <!-- Full Screen Result View (Comparison Slider) -->
            <div id="vton-result-view" style="display: none; padding: 24px; border-top: 1px solid var(--vton-border);">
              <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <span style="font-weight: 700; color: #111827;">Interactive Comparison (Slide to compare)</span>
                <span id="vton-delta-badge" class="vton-badge" style="display:none; background: #ecfdf5; color: #059669;">
                  ✓ Kurta Flaps Automatically Removed
                </span>
              </div>
              <div class="vton-comparison-container" id="vton-slider-container">
                <img id="vton-cmp-before" />
                <div class="vton-comparison-after" id="vton-cmp-after-box">
                  <img id="vton-cmp-after" />
                </div>
                <div class="vton-slider-handle" id="vton-slider-handle"></div>
              </div>
            </div>

          </div>
        </div>
      `;

      const div = document.createElement('div');
      div.innerHTML = modalHtml;
      document.body.appendChild(div);

      this.bindEvents();
    }

    bindEvents() {
      const backdrop = document.getElementById('vton-modal-root');
      const closeBtn = document.getElementById('vton-close-btn');
      const dropzone = document.getElementById('vton-dropzone');
      const fileInput = document.getElementById('vton-file-input');
      const sampleBtn = document.getElementById('vton-sample-kurta-btn');
      const submitBtn = document.getElementById('vton-submit-btn');

      closeBtn.addEventListener('click', () => this.close());
      backdrop.addEventListener('click', (e) => {
        if (e.target === backdrop) this.close();
      });

      dropzone.addEventListener('click', () => fileInput.click());
      fileInput.addEventListener('change', (e) => {
        if (e.target.files && e.target.files[0]) {
          this.loadFile(e.target.files[0]);
        }
      });

      sampleBtn.addEventListener('click', () => {
        this.loadSampleKurtaPhoto();
      });

      submitBtn.addEventListener('click', () => {
        this.startTryOn();
      });

      this.initSlider();
    }

    open(options = {}) {
      this.activeGarment = options.garmentUrl || null;
      const garmentImg = document.getElementById('vton-garment-preview');
      if (this.activeGarment) {
        garmentImg.src = this.activeGarment;
      }
      const backdrop = document.getElementById('vton-modal-root');
      backdrop.classList.add('active');
    }

    close() {
      const backdrop = document.getElementById('vton-modal-root');
      backdrop.classList.remove('active');
      if (this.eventSource) {
        this.eventSource.close();
        this.eventSource = null;
      }
    }

    loadFile(file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        this.selectedUserImageB64 = e.target.result;
        const preview = document.getElementById('vton-user-preview');
        const prompt = document.getElementById('vton-upload-prompt');
        preview.src = this.selectedUserImageB64;
        preview.style.display = 'block';
        prompt.style.display = 'none';
        document.getElementById('vton-submit-btn').disabled = false;
      };
      reader.readAsDataURL(file);
    }

    loadSampleKurtaPhoto() {
      // Creates an inline synthetic representation of a person wearing a knee-length kurta
      const canvas = document.createElement('canvas');
      canvas.width = 600;
      canvas.height = 900;
      const ctx = canvas.getContext('2d');

      // Studio background
      ctx.fillStyle = '#f1f5f9';
      ctx.fillRect(0, 0, 600, 900);

      // Head & Face
      ctx.fillStyle = '#d4a373';
      ctx.beginPath();
      ctx.arc(300, 120, 60, 0, Math.PI * 2);
      ctx.fill();

      // Hair
      ctx.fillStyle = '#1e293b';
      ctx.beginPath();
      ctx.arc(300, 90, 60, Math.PI, Math.PI * 2);
      ctx.fill();

      // Long Kurta (Shoulders down to knees, y=180 to y=640)
      ctx.fillStyle = '#0284c7'; // Rich Sky Blue Kurta
      ctx.beginPath();
      ctx.moveTo(200, 190);
      ctx.lineTo(400, 190);
      ctx.lineTo(420, 640); // Hem at knee level
      ctx.lineTo(180, 640);
      ctx.closePath();
      ctx.fill();

      // Arms
      ctx.fillRect(140, 190, 60, 240);
      ctx.fillRect(400, 190, 60, 240);

      // Lower Shalwar / Pajama visible below the knee (y=640 to y=820)
      ctx.fillStyle = '#e2e8f0'; // Off-white Shalwar
      ctx.fillRect(210, 640, 75, 180);
      ctx.fillRect(315, 640, 75, 180);

      // Shoes
      ctx.fillStyle = '#451a03';
      ctx.fillRect(200, 820, 85, 30);
      ctx.fillRect(315, 820, 85, 30);

      const b64 = canvas.toDataURL('image/jpeg');
      this.selectedUserImageB64 = b64;
      const preview = document.getElementById('vton-user-preview');
      const prompt = document.getElementById('vton-upload-prompt');
      preview.src = b64;
      preview.style.display = 'block';
      prompt.style.display = 'none';
      document.getElementById('vton-submit-btn').disabled = false;
    }

    async startTryOn() {
      if (this.isProcessing) return;
      this.isProcessing = true;

      const submitBtn = document.getElementById('vton-submit-btn');
      const progressBox = document.getElementById('vton-progress-box');
      const resultView = document.getElementById('vton-result-view');

      submitBtn.disabled = true;
      progressBox.style.display = 'flex';
      resultView.style.display = 'none';

      try {
        const isGarmentB64 = this.activeGarment && this.activeGarment.startsWith("data:");
        const payload = {
          store_id: this.storeId,
          user_image_base64: this.selectedUserImageB64,
          garment_image_url: isGarmentB64 ? null : this.activeGarment,
          garment_image_base64: isGarmentB64 ? this.activeGarment : null,
          garment_category: "upper_body",
          options: {
            preserve_face: true,
            handle_long_garment_delta: true,
            auto_reconstruct_lower_wear: true
          }
        };

        const res = await fetch(`${this.apiBase}/api/v1/tryon/jobs`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });

        if (!res.ok) throw new Error("Failed to initialize try-on job.");
        const job = await res.json();
        this.trackJob(job.job_id);

      } catch (err) {
        alert("Virtual Try-On error: " + err.message);
        this.isProcessing = false;
        submitBtn.disabled = false;
        progressBox.style.display = 'none';
      }
    }

    trackJob(jobId) {
      // Listen to SSE progress stream
      this.eventSource = new EventSource(`${this.apiBase}/api/v1/tryon/jobs/${jobId}/stream`);

      this.eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);
        this.updateProgress(data.stage, data.progress_percentage);

        if (data.status === "completed") {
          this.eventSource.close();
          this.eventSource = null;
          this.showResult(data);
        } else if (data.status === "failed") {
          this.eventSource.close();
          this.eventSource = null;
          alert("Try-On processing failed: " + (data.error || "Unknown error"));
          this.resetButtons();
        }
      };

      this.eventSource.onerror = () => {
        // Fallback polling if SSE is disconnected
        this.pollJob(jobId);
      };
    }

    async pollJob(jobId) {
      const interval = setInterval(async () => {
        try {
          const res = await fetch(`${this.apiBase}/api/v1/tryon/jobs/${jobId}`);
          const data = await res.json();
          this.updateProgress(data.stage, data.progress_percentage);

          if (data.status === "completed") {
            clearInterval(interval);
            this.showResult(data);
          } else if (data.status === "failed") {
            clearInterval(interval);
            alert("Try-On failed: " + data.error);
            this.resetButtons();
          }
        } catch (e) {
          clearInterval(interval);
        }
      }, 1500);
    }

    updateProgress(stage, pct) {
      const fill = document.getElementById('vton-progress-fill');
      const text = document.getElementById('vton-stage-text');
      const pctLabel = document.getElementById('vton-stage-pct');

      fill.style.width = `${pct}%`;
      pctLabel.innerText = `${pct}%`;
      text.innerText = STAGE_LABELS[stage] || stage;
    }

    showResult(jobData) {
      this.resetButtons();
      const resultView = document.getElementById('vton-result-view');
      const cmpBefore = document.getElementById('vton-cmp-before');
      const cmpAfter = document.getElementById('vton-cmp-after');
      const deltaBadge = document.getElementById('vton-delta-badge');

      cmpBefore.src = this.selectedUserImageB64;
      cmpAfter.src = jobData.result_image_url;

      deltaBadge.style.display = 'inline-block';
      resultView.style.display = 'block';

      // Scroll to result
      resultView.scrollIntoView({ behavior: 'smooth' });
    }

    resetButtons() {
      this.isProcessing = false;
      document.getElementById('vton-submit-btn').disabled = false;
      document.getElementById('vton-progress-box').style.display = 'none';
    }

    initSlider() {
      const container = document.getElementById('vton-slider-container');
      const afterBox = document.getElementById('vton-cmp-after-box');
      const handle = document.getElementById('vton-slider-handle');

      let isDragging = false;

      const setPosition = (clientX) => {
        const rect = container.getBoundingClientRect();
        let pos = (clientX - rect.left) / rect.width;
        pos = Math.max(0, Math.min(1, pos));
        afterBox.style.width = `${pos * 100}%`;
        handle.style.left = `${pos * 100}%`;
      };

      container.addEventListener('mousedown', (e) => {
        isDragging = true;
        setPosition(e.clientX);
      });
      window.addEventListener('mouseup', () => { isDragging = false; });
      window.addEventListener('mousemove', (e) => {
        if (!isDragging) return;
        setPosition(e.clientX);
      });

      // Touch events for mobile
      container.addEventListener('touchstart', (e) => {
        isDragging = true;
        setPosition(e.touches[0].clientX);
      });
      window.addEventListener('touchend', () => { isDragging = false; });
      window.addEventListener('touchmove', (e) => {
        if (!isDragging) return;
        setPosition(e.touches[0].clientX);
      });
    }
  }

  window.VirtualTryOn = new VirtualTryOnClient();
})(window);

