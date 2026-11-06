import os
import uuid
import base64
import httpx
from io import BytesIO
from PIL import Image
from typing import Tuple, Optional
from api.config import settings

class StorageService:
    def __init__(self):
        self.backend = settings.STORAGE_BACKEND
        if self.backend == "local":
            os.makedirs(settings.LOCAL_STORAGE_DIR, exist_ok=True)
            self.uploads_dir = os.path.join(settings.LOCAL_STORAGE_DIR, "uploads")
            self.renders_dir = os.path.join(settings.LOCAL_STORAGE_DIR, "renders")
            os.makedirs(self.uploads_dir, exist_ok=True)
            os.makedirs(self.renders_dir, exist_ok=True)
        else:
            # S3 / Cloudflare R2 client
            import boto3
            self.s3_client = boto3.client(
                "s3",
                endpoint_url=settings.S3_ENDPOINT_URL or None,
                aws_access_key_id=settings.S3_ACCESS_KEY_ID,
                aws_secret_access_key=settings.S3_SECRET_ACCESS_KEY,
            )
            self.bucket_name = settings.S3_BUCKET_NAME

    async def fetch_image(self, url: Optional[str] = None, b64_data: Optional[str] = None) -> Image.Image:
        """Fetch image from a remote HTTP URL, Base64 string, or data URI."""
        if b64_data:
            if "," in b64_data:
                b64_data = b64_data.split(",", 1)[1]
            image_bytes = base64.b64decode(b64_data)
            return Image.open(BytesIO(image_bytes)).convert("RGB")
        
        if url:
            # Handle inline Data URI (e.g. data:image/png;base64,...)
            if url.startswith("data:"):
                b64_content = url.split(",", 1)[1] if "," in url else url
                image_bytes = base64.b64decode(b64_content)
                return Image.open(BytesIO(image_bytes)).convert("RGB")

            # Handle local file paths
            if url.startswith("file://") or os.path.isabs(url) or os.path.exists(url):
                local_path = url.replace("file://", "")
                return Image.open(local_path).convert("RGB")

            # Handle relative local storage paths
            if url.startswith("/static/"):
                clean_rel = url.replace("/static/", "").replace("/", os.sep)
                local_path = os.path.join(settings.LOCAL_STORAGE_DIR, clean_rel)
                if os.path.exists(local_path):
                    return Image.open(local_path).convert("RGB")

            # Remote HTTP / HTTPS URL
            async with httpx.AsyncClient(timeout=30.0) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                return Image.open(BytesIO(resp.content)).convert("RGB")
                
        raise ValueError("Neither url nor base64 image data was provided.")

    def save_image(self, image: Image.Image, category: str = "renders", ext: str = "webp") -> Tuple[str, str]:
        """Save a PIL Image locally or to S3/R2. Returns (storage_key_or_path, public_url)."""
        filename = f"{uuid.uuid4().hex}.{ext}"
        
        if self.backend == "local":
            target_dir = self.renders_dir if category == "renders" else self.uploads_dir
            file_path = os.path.join(target_dir, filename)
            
            # Save optimized
            if ext == "webp":
                image.save(file_path, "WEBP", quality=92)
            else:
                image.save(file_path, quality=95)
                
            public_url = f"/static/{category}/{filename}"
            return file_path, public_url
        else:
            # S3 / Cloudflare R2
            buffer = BytesIO()
            if ext == "webp":
                image.save(buffer, format="WEBP", quality=92)
                content_type = "image/webp"
            else:
                image.save(buffer, format="JPEG", quality=95)
                content_type = "image/jpeg"
                
            buffer.seek(0)
            s3_key = f"{category}/{filename}"
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=s3_key,
                Body=buffer,
                ContentType=content_type
            )
            public_url = f"{settings.S3_PUBLIC_DOMAIN.rstrip('/')}/{s3_key}"
            return s3_key, public_url

storage_service = StorageService()

