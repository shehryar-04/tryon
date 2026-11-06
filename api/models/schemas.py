from enum import Enum
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, HttpUrl
from datetime import datetime

class GarmentCategory(str, Enum):
    UPPER_BODY = "upper_body"        # Shirts, t-shirts, polo, blouses, sweaters
    LOWER_BODY = "lower_body"        # Pants, jeans, shorts, skirts
    DRESS = "dress"                  # Dresses, gowns, full-length kurtas, thobes
    OUTERWEAR = "outerwear"          # Jackets, blazers, coats

class QualityPreset(str, Enum):
    HIGH_FIDELITY = "high_fidelity"  # Leffa / IDM-VTON (Flow alignment, maximum micro-texture)
    FAST = "fast"                    # CatVTON (Concatenation-based, ultra-fast latency)

class LowerWearReconstruction(str, Enum):
    AUTO_PROPAGATE = "auto_propagate" # Propagate visible fabric (shalwar/trousers) upward
    NEUTRAL_DENIM = "neutral_denim"   # Synthesize dark indigo jeans
    NEUTRAL_CHINOS = "neutral_chinos" # Synthesize slim khaki/beige chinos
    NONE = "none"

class TryOnJobOptions(BaseModel):
    preserve_face: bool = Field(default=True, description="Strictly preserve original facial features and hair")
    handle_long_garment_delta: bool = Field(
        default=True,
        description="Detect if user wears a long garment (kurta/dress) and target is shorter (shirt); automatically removes lower flaps and reconstructs trousers"
    )
    auto_reconstruct_lower_wear: bool = Field(default=True, description="Inpaint pants if covered by source long garment")
    lower_wear_style: LowerWearReconstruction = Field(default=LowerWearReconstruction.AUTO_PROPAGATE)
    quality_preset: QualityPreset = Field(default=QualityPreset.HIGH_FIDELITY)
    guidance_scale: float = Field(default=2.5, ge=1.0, le=10.0, description="Classifier-free guidance strength")
    num_inference_steps: int = Field(default=30, ge=10, le=50, description="Denoising steps")

class CreateTryOnJobRequest(BaseModel):
    store_id: str = Field(..., description="Unique e-commerce store ID")
    user_image_url: Optional[str] = Field(default=None, description="Public URL of shopper's photo")
    user_image_base64: Optional[str] = Field(default=None, description="Base64 encoded shopper's photo")
    garment_image_url: Optional[str] = Field(default=None, description="Public URL of clothing product image")
    garment_image_base64: Optional[str] = Field(default=None, description="Base64 encoded clothing product image")
    garment_category: GarmentCategory = Field(default=GarmentCategory.UPPER_BODY)
    options: TryOnJobOptions = Field(default_factory=TryOnJobOptions)
    webhook_url: Optional[str] = Field(default=None, description="Store webhook called upon completion")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Custom store metadata passed back in webhook")

class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class JobStage(str, Enum):
    QUEUED = "queued"
    PARSING = "parsing"                 # Stage 1: Grounded-SAM 2 & DensePose
    PRE_INPAINT = "pre_inpaint"         # Stage 2: Kurta delta removal & LaMa pants reconstruction
    DIFFUSION = "diffusion"             # Stage 3: Leffa / IDM-VTON try-on synthesis
    BLENDING = "blending"               # Stage 4: Face identity preservation & Poisson blend
    COMPLETED = "completed"
    FAILED = "failed"

class TryOnJobResponse(BaseModel):
    job_id: str
    store_id: str
    status: JobStatus
    stage: JobStage
    progress_percentage: int = Field(ge=0, le=100)
    result_image_url: Optional[str] = None
    execution_time_ms: Optional[float] = None
    error: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    metadata: Optional[Dict[str, Any]] = None

class WebhookPayload(BaseModel):
    event: str = "tryon.completed"
    job_id: str
    store_id: str
    status: JobStatus
    result_image_url: Optional[str] = None
    execution_time_ms: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)

