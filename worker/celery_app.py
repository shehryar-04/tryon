from celery import Celery
from api.config import settings

celery_app = Celery(
    "vton_worker",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=["worker.tasks"]
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=300,        # 5 minutes max per try-on generation
    worker_prefetch_multiplier=1, # 1 task per GPU worker at a time for maximum throughput
)

