import time
import httpx
import asyncio
from typing import Dict, Any
from worker.celery_app import celery_app
from api.services.storage import storage_service
from engine.pipeline import VirtualTryOnPipeline
from api.models.schemas import JobStatus, JobStage

# Global pipeline instance per worker process
_pipeline_instance = None

def get_pipeline():
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = VirtualTryOnPipeline(model_preset="leffa")
    return _pipeline_instance

@celery_app.task(bind=True, name="worker.tasks.execute_tryon_task")
def execute_tryon_task(self, job_payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Celery GPU task executing the 4-stage Virtual Try-On generation.
    """
    job_id = job_payload["job_id"]
    store_id = job_payload["store_id"]
    webhook_url = job_payload.get("webhook_url")
    options = job_payload.get("options", {})
    garment_category = job_payload.get("garment_category", "upper_body")

    def update_progress(stage: str, percentage: int):
        self.update_state(
            state="PROGRESS",
            meta={
                "stage": stage,
                "progress_percentage": percentage,
                "job_id": job_id
            }
        )

    try:
        update_progress(JobStage.PARSING.value, 10)
        
        # 1. Fetch input images
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        user_img = loop.run_until_complete(
            storage_service.fetch_image(
                url=job_payload.get("user_image_url"),
                b64_data=job_payload.get("user_image_base64")
            )
        )
        garment_img = loop.run_until_complete(
            storage_service.fetch_image(
                url=job_payload.get("garment_image_url"),
                b64_data=job_payload.get("garment_image_base64")
            )
        )

        # 2. Run the 4-stage engine
        pipeline = get_pipeline()
        result = pipeline.run(
            person_image=user_img,
            garment_image=garment_img,
            garment_category=garment_category,
            options=options,
            progress_callback=update_progress
        )

        # 3. Save result image
        final_img = result["result_image"]
        storage_path, public_url = storage_service.save_image(final_img, category="renders", ext="webp")

        output_payload = {
            "job_id": job_id,
            "store_id": store_id,
            "status": JobStatus.COMPLETED.value,
            "stage": JobStage.COMPLETED.value,
            "progress_percentage": 100,
            "result_image_url": public_url,
            "execution_time_ms": result["execution_time_ms"],
            "delta_detected": result["delta_detected"],
            "detected_source_garment": result["detected_source_garment"]
        }

        # 4. Dispatch Webhook to Store if requested
        if webhook_url:
            try:
                with httpx.Client(timeout=10.0) as client:
                    client.post(webhook_url, json={
                        "event": "tryon.completed",
                        "data": output_payload,
                        "metadata": job_payload.get("metadata", {})
                    })
            except Exception as e:
                print(f"[!] Warning: Webhook dispatch to {webhook_url} failed: {e}")

        return output_payload

    except Exception as exc:
        self.update_state(
            state="FAILURE",
            meta={"stage": JobStage.FAILED.value, "error": str(exc), "job_id": job_id}
        )
        raise exc

