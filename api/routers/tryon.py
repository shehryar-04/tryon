import json
import asyncio
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File, Form
from sse_starlette.sse import EventSourceResponse
from typing import Optional

from api.models.schemas import (
    CreateTryOnJobRequest,
    TryOnJobResponse,
    JobStatus,
    GarmentCategory,
    TryOnJobOptions
)
from api.services.queue import queue_service
from api.services.storage import storage_service

router = APIRouter(prefix="/tryon", tags=["Virtual Try-On"])

@router.post("/jobs", response_model=TryOnJobResponse, status_code=202)
async def create_tryon_job(request: CreateTryOnJobRequest):
    """
    Submits an asynchronous Virtual Try-On job.
    Accepts image URLs or Base64 encoded images.
    Returns immediately with a job_id and status='queued'.
    """
    if not (request.user_image_url or request.user_image_base64):
        raise HTTPException(status_code=400, detail="Shopper photo must be provided as URL or base64.")
    if not (request.garment_image_url or request.garment_image_base64):
        raise HTTPException(status_code=400, detail="Garment image must be provided as URL or base64.")

    job = await queue_service.submit_job(request)
    return job

@router.post("/jobs/upload", response_model=TryOnJobResponse, status_code=202)
async def create_tryon_job_upload(
    store_id: str = Form(...),
    garment_category: str = Form("upper_body"),
    preserve_face: bool = Form(True),
    handle_long_garment_delta: bool = Form(True),
    webhook_url: Optional[str] = Form(None),
    user_file: UploadFile = File(...),
    garment_file: UploadFile = File(...)
):
    """
    Direct multipart/form-data upload endpoint.
    Ideal for direct mobile / storefront HTML form submissions.
    """
    user_bytes = await user_file.read()
    garment_bytes = await garment_file.read()
    
    import base64
    user_b64 = base64.b64encode(user_bytes).decode("utf-8")
    garment_b64 = base64.b64encode(garment_bytes).decode("utf-8")

    options = TryOnJobOptions(
        preserve_face=preserve_face,
        handle_long_garment_delta=handle_long_garment_delta
    )
    
    req = CreateTryOnJobRequest(
        store_id=store_id,
        user_image_base64=user_b64,
        garment_image_base64=garment_b64,
        garment_category=GarmentCategory(garment_category),
        options=options,
        webhook_url=webhook_url
    )
    return await queue_service.submit_job(req)

@router.get("/jobs/{job_id}", response_model=TryOnJobResponse)
async def get_job_status(job_id: str):
    """Poll the status and progress of a try-on job."""
    job = await queue_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return job

@router.get("/jobs/{job_id}/stream")
async def stream_job_progress(job_id: str):
    """
    Server-Sent Events (SSE) endpoint.
    Emits real-time progress updates (stage, % completed, result URL) to the browser.
    """
    job = await queue_service.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    async def event_generator():
        # First send current state
        data_str = job.model_dump_json() if hasattr(job, "model_dump_json") else job.json()
        yield {
            "event": "progress",
            "data": data_str
        }
        
        if job.status in [JobStatus.COMPLETED, JobStatus.FAILED]:
            return

        queue = queue_service.subscribe_progress(job_id)
        try:
            while True:
                data = await queue.get()
                parsed = json.loads(data)
                yield {
                    "event": "progress",
                    "data": data
                }
                if parsed.get("status") in [JobStatus.COMPLETED.value, JobStatus.FAILED.value]:
                    break
        finally:
            queue_service.unsubscribe_progress(job_id, queue)

    return EventSourceResponse(event_generator())

