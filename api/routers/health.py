import os
from fastapi import APIRouter
from api.config import settings

router = APIRouter(tags=["Health & Status"])

@router.get("/health")
async def health_check():
    """Returns system status, active device, and memory statistics."""
    cuda_available = False
    device_name = "CPU"
    vram_free_gb = 0.0

    try:
        import torch
        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            mem_info = torch.cuda.mem_get_info()
            vram_free_gb = round(mem_info[0] / (1024**3), 2)
    except ImportError:
        pass

    return {
        "status": "healthy",
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "device": device_name,
        "cuda_available": cuda_available,
        "free_vram_gb": vram_free_gb,
        "default_tryon_model": settings.DEFAULT_TRYON_MODEL,
        "storage_backend": settings.STORAGE_BACKEND
    }

