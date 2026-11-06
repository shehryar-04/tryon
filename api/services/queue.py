import asyncio
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
from api.models.schemas import (
    CreateTryOnJobRequest,
    TryOnJobResponse,
    JobStatus,
    JobStage
)
from api.services.storage import storage_service
from engine.pipeline import VirtualTryOnPipeline
from api.config import settings
import httpx

# In-memory job registry for instant local querying and dev fallback
_job_store: Dict[str, Dict[str, Any]] = {}
_sse_subscribers: Dict[str, list] = {}

class QueueService:
    def __init__(self):
        self._pipeline = None

    def _get_pipeline(self):
        if self._pipeline is None:
            self._pipeline = VirtualTryOnPipeline(model_preset="leffa")
        return self._pipeline

    async def submit_job(self, req: CreateTryOnJobRequest) -> TryOnJobResponse:
        job_id = f"job_{uuid.uuid4().hex[:12]}"
        now = datetime.utcnow()
        
        req_data = req.model_dump() if hasattr(req, "model_dump") else req.dict()
        job_entry = {
            "job_id": job_id,
            "store_id": req.store_id,
            "status": JobStatus.QUEUED,
            "stage": JobStage.QUEUED,
            "progress_percentage": 0,
            "result_image_url": None,
            "execution_time_ms": None,
            "error": None,
            "created_at": now,
            "updated_at": now,
            "metadata": req.metadata,
            "request_data": req_data
        }
        _job_store[job_id] = job_entry

        # Dispatch to Celery if configured, otherwise execute locally
        if settings.USE_CELERY:
            try:
                from worker.tasks import execute_tryon_task
                payload = dict(req_data)
                payload["job_id"] = job_id
                celery_task = execute_tryon_task.delay(payload)
                job_entry["celery_task_id"] = celery_task.id
            except Exception as e:
                print(f"[!] Celery dispatch failed: {e}, running locally")
                asyncio.create_task(self._run_local_job(job_id, req))
        else:
            # Instant local asynchronous task execution
            asyncio.create_task(self._run_local_job(job_id, req))

        return TryOnJobResponse(**job_entry)

    async def _run_local_job(self, job_id: str, req: CreateTryOnJobRequest):
        entry = _job_store[job_id]
        entry["status"] = JobStatus.PROCESSING
        entry["stage"] = JobStage.PARSING
        entry["progress_percentage"] = 15
        entry["updated_at"] = datetime.utcnow()
        await self._broadcast_progress(job_id)

        try:
            user_img = await storage_service.fetch_image(
                url=req.user_image_url,
                b64_data=req.user_image_base64
            )
            garment_img = await storage_service.fetch_image(
                url=req.garment_image_url,
                b64_data=req.garment_image_base64
            )

            loop = asyncio.get_running_loop()

            def progress_hook(stage: str, pct: int):
                entry["stage"] = JobStage(stage)
                entry["progress_percentage"] = pct
                entry["updated_at"] = datetime.utcnow()
                asyncio.run_coroutine_threadsafe(
                    self._broadcast_progress(job_id),
                    loop
                )

            pipeline = self._get_pipeline()
            opt_dict = req.options.model_dump() if hasattr(req.options, "model_dump") else req.options.dict()
            
            # Run pipeline in threadpool to keep FastAPI async event loop responsive
            result = await loop.run_in_executor(
                None,
                lambda: pipeline.run(
                    person_image=user_img,
                    garment_image=garment_img,
                    garment_category=req.garment_category.value,
                    options=opt_dict,
                    progress_callback=progress_hook
                )
            )

            # Save result image
            path, public_url = storage_service.save_image(result["result_image"], category="renders", ext="webp")

            entry["status"] = JobStatus.COMPLETED
            entry["stage"] = JobStage.COMPLETED
            entry["progress_percentage"] = 100
            entry["result_image_url"] = public_url
            entry["execution_time_ms"] = result["execution_time_ms"]
            entry["updated_at"] = datetime.utcnow()
            await self._broadcast_progress(job_id)

            # Webhook dispatch
            if req.webhook_url:
                try:
                    async with httpx.AsyncClient(timeout=10.0) as client:
                        await client.post(req.webhook_url, json={
                            "event": "tryon.completed",
                            "data": {
                                "job_id": job_id,
                                "store_id": req.store_id,
                                "status": "completed",
                                "result_image_url": public_url,
                                "execution_time_ms": result["execution_time_ms"],
                                "delta_detected": result["delta_detected"]
                            },
                            "metadata": req.metadata
                        })
                except Exception as e:
                    print(f"[!] Webhook error: {e}")

        except Exception as e:
            entry["status"] = JobStatus.FAILED
            entry["stage"] = JobStage.FAILED
            entry["error"] = str(e)
            entry["updated_at"] = datetime.utcnow()
            await self._broadcast_progress(job_id)

    async def get_job(self, job_id: str) -> Optional[TryOnJobResponse]:
        if job_id not in _job_store:
            return None
        entry = _job_store[job_id]
        return TryOnJobResponse(**entry)

    async def _broadcast_progress(self, job_id: str):
        if job_id in _sse_subscribers:
            entry = _job_store[job_id]
            resp_obj = TryOnJobResponse(**entry)
            data = resp_obj.model_dump_json() if hasattr(resp_obj, "model_dump_json") else resp_obj.json()
            for q in list(_sse_subscribers[job_id]):
                await q.put(data)

    def subscribe_progress(self, job_id: str) -> asyncio.Queue:
        if job_id not in _sse_subscribers:
            _sse_subscribers[job_id] = []
        q = asyncio.Queue()
        _sse_subscribers[job_id].append(q)
        return q

    def unsubscribe_progress(self, job_id: str, q: asyncio.Queue):
        if job_id in _sse_subscribers and q in _sse_subscribers[job_id]:
            _sse_subscribers[job_id].remove(q)

queue_service = QueueService()

