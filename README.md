# Virtual Try-On API Platform for E-Commerce

A production-ready, open-source AI Virtual Try-On platform designed for seamless integration with e-commerce clothing stores (Shopify, WooCommerce, custom storefronts). Shoppers upload their photo on the merchant's store, choose an apparel product, and our GPU backend synthesizes a photorealistic virtual try-on in real time.

---

## 🌟 Key Highlights & The "Kurta-to-Shirt" Formula

Most off-the-shelf virtual try-on models fail when switching between garments of different lengths—for instance, when a shopper wearing a knee-length **Kurta, Thobe, or Dress** attempts to try on a standard **Shirt or T-Shirt**:
- **Naive VTON Failure (Ghost Flaps)**: Standard upper-body masks only cover down to the beltline. The bottom half of the kurta (waist to knees) remains untouched, creating an absurd output where a shirt is pasted onto the chest while the kurta's bottom flaps still drape over the legs.
- **Naive VTON Failure (Lower Voids)**: If you blindly mask the entire kurta down to the knees, an upper-body model does not know what to generate between the hips and knees, causing warped voids or mutated limbs.

### The 4-Stage Compositional Solution:
1. **Stage 1 (Open-Vocabulary Segmentation & Pose)**:
   - Uses **Grounded-SAM 2** (Florence-2 / Grounding DINO + SAM 2) and **DensePose / DWPose** to detect the full boundary of the source garment ($Y_{top}$ to $Y_{bottom}$) and 3D body keypoints (neck, shoulders, waistline $Y_{waist}$, knees $Y_{knee}$, and ankles).
2. **Stage 2 (Garment Delta Removal & Lower-Body Inpainting)**:
   - Evaluates the discrepancy between the long garment hem and the target shirt hem $Y_{shirt\_hem}$.
   - Extracts the **Delta Region**: $M_{\Delta} = M_{source} \cap (y > Y_{shirt\_hem})$.
   - Erases the kurta flaps in $M_{\Delta}$ and reconstructs the trousers/shalwar/pants upward to the natural waistline using **LaMa (Large Mask Inpainting)** fast Fourier convolutions.
3. **Stage 3 (SOTA Flow-Guided Diffusion Try-On)**:
   - Feeds the clean agnostic torso and restored lower-body canvas into **Leffa (CVPR 2025)** or **IDM-VTON (ECCV 2024)** (or lightweight **CatVTON** for <8GB VRAM).
   - Generates photorealistic fabric drape, collars, buttons, and natural fit right over the restored waistband.
4. **Stage 4 (Identity & Detail Blending)**:
   - Strictly preserves the original face, hair, wrists, and background using **CodeFormer** and Poisson seamless blending.

---

## 🛠️ Open-Source Stack Composition

| Role | Primary Open-Source Project | Advantage |
| :--- | :--- | :--- |
| **Try-On Diffusion Engine** | [Leffa (CVPR 2025)](https://github.com/franciszzj/Leffa) | Attention flow fields; prevents fabric texture distortion. |
| **High-Pattern Fidelity** | [IDM-VTON (ECCV 2024)](https://github.com/yisol/IDM-VTON) | Dual-path GarmentNet + cross-attention IP-Adapter. |
| **Budget / High-Throughput** | [CatVTON (ICLR 2025)](https://github.com/Zheng-Chong/CatVTON) | Lightweight concatenation latent diffusion (<8GB VRAM). |
| **Garment Segmentation** | [Segment Anything 2 (SAM 2)](https://github.com/facebookresearch/segment-anything-2) | Open-vocabulary text-prompted segmentation. |
| **Body Keypoints & 3D Surface** | [DensePose](https://github.com/facebookresearch/detectron2/tree/main/projects/DensePose) + DWPose | 3D IUV surface mapping & whole-body skeletal landmarks. |
| **Delta Pre-Inpainting** | [LaMa Inpainting](https://github.com/advimman/lama) | Sub-second fast Fourier convolutions for pants reconstruction. |
| **Detail Restoration** | [CodeFormer](https://github.com/sczhou/CodeFormer) | Preserves shopper facial identity and micro-features. |
| **API Gateway** | [FastAPI](https://fastapi.tiangolo.com/) | High-speed async REST endpoints with Swagger & SSE streaming. |
| **Distributed Task Queue** | [Celery](https://docs.celeryq.dev/) + [Redis](https://redis.io/) | Asynchronous GPU job scheduling and cluster orchestration. |

---

## 📁 Project Structure

```
d:\tryon/
├── api/                          # FastAPI REST API Gateway
│   ├── main.py                   # Entrypoint & route registration
│   ├── config.py                 # App settings & storage configuration
│   ├── models/schemas.py         # Pydantic schemas (JobRequest, JobResponse, Webhooks)
│   ├── routers/
│   │   ├── tryon.py              # /api/v1/tryon endpoints (submit, poll, SSE stream)
│   │   └── health.py             # Health checks & GPU metrics
│   └── services/
│       ├── storage.py            # Local / Cloudflare R2 / S3 image manager
│       └── queue.py              # Task dispatcher & SSE event broadcaster
│
├── engine/                       # Core AI Try-On Engine
│   ├── pipeline.py               # Master 4-stage pipeline orchestrator
│   ├── stages/
│   │   ├── pose.py               # Stage 1: DensePose & DWPose landmarks
│   │   ├── segmenter.py          # Stage 1: SAM 2 open-vocabulary parser
│   │   ├── pre_inpaint.py        # Stage 2: Kurta delta removal & LaMa pants inpainting
│   │   ├── vton.py               # Stage 3: Leffa / IDM-VTON / CatVTON adapter
│   │   └── blender.py            # Stage 4: CodeFormer & Poisson boundary blending
│   └── weights/
│       └── download_models.py    # Automated checkpoint downloader from Hugging Face
│
├── worker/                       # Celery GPU Worker Daemon
│   ├── celery_app.py             # Celery worker initialization
│   └── tasks.py                  # Distributed async execution task & store webhooks
│
├── widget/                       # Storefront Integration SDK
│   ├── embed.js                  # Drop-in JavaScript widget for Shopify / WooCommerce
│   ├── index.html                # Interactive test storefront demo ("Aurora Couture")
│   └── widget.css                # Polished modal & split-comparison slider styles
│
├── tests/                        # Automated Test Suite
│   ├── test_kurta_delta.py       # Acid test verifying kurta flap removal & pants inpainting
│   ├── test_pipeline_unit.py     # 4-stage pipeline execution test
│   └── test_api.py               # FastAPI schema & endpoint validation tests
│
├── docker-compose.yml            # Multi-service container orchestration
├── Dockerfile.gpu                # CUDA 12.1+ PyTorch base image for GPU workers
└── requirements.txt              # Pinned production dependencies
```

---

## 🚀 Quick Start Guide

### 1. Local Run with Python / Uvicorn
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start the FastAPI API Server
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```
Once started, visit:
- **Interactive Storefront Demo**: `http://localhost:8000/widget/index.html`
- **Swagger API Documentation**: `http://localhost:8000/docs`

### 2. Download Model Checkpoints
```bash
# Download Leffa (CVPR 2025)
python engine/weights/download_models.py --model leffa

# Or download CatVTON for lightweight GPUs (<8GB VRAM)
python engine/weights/download_models.py --model cat_vton
```

### 3. GPU Worker Cluster with Docker
```bash
docker-compose up --build
```

---

## 🔌 E-Commerce Integration (Shopify, WooCommerce, Custom)

### Option A: 2-Line Storefront JavaScript Widget
Add this snippet to your store theme (e.g. `theme.liquid` in Shopify):
```html
<link rel="stylesheet" href="https://api.your-tryon-domain.com/widget/widget.css">
<script src="https://api.your-tryon-domain.com/widget/embed.js"></script>

<!-- Add a Try-On button to your Product Page -->
<button onclick="window.VirtualTryOn.open({
  garmentUrl: '{{ product.featured_image | img_url: 'master' }}',
  garmentCategory: 'upper_body',
  storeId: 'your_shopify_store_id'
})">
  ✨ Try On Me
</button>
```

### Option B: Headless REST API
Submit a job programmatically from your backend:
```bash
curl -X POST "https://api.your-tryon-domain.com/api/v1/tryon/jobs" \
  -H "Content-Type: application/json" \
  -d '{
    "store_id": "store_aurora",
    "user_image_url": "https://cdn.store.com/customer_photo.jpg",
    "garment_image_url": "https://cdn.store.com/oxford_shirt.jpg",
    "garment_category": "upper_body",
    "options": {
      "preserve_face": true,
      "handle_long_garment_delta": true,
      "auto_reconstruct_lower_wear": true
    },
    "webhook_url": "https://store.com/api/webhooks/tryon-result"
  }'
```

---

## 🧪 Verification & Running Tests
Run the automated test suite:
```bash
# Verify Kurta-to-Shirt garment removal & pants inpainting
pytest tests/test_kurta_delta.py -v

# Verify full 4-stage pipeline execution
pytest tests/test_pipeline_unit.py -v

# Verify API contracts
pytest tests/test_api.py -v
```

