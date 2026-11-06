import pytest
from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_check_endpoint():
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "device" in data
    assert "version" in data

def test_create_tryon_job_validation_error():
    # Sending empty payload must fail with 422
    response = client.post("/api/v1/tryon/jobs", json={})
    assert response.status_code == 422

def test_create_tryon_job_missing_images():
    # Missing images must return 400
    payload = {
        "store_id": "store_aurora_test",
        "garment_category": "upper_body"
    }
    response = client.post("/api/v1/tryon/jobs", json=payload)
    assert response.status_code == 400

def test_create_tryon_job_success():
    # Valid submission with mock base64 images
    payload = {
        "store_id": "store_aurora_test",
        "user_image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
        "garment_image_base64": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==",
        "garment_category": "upper_body",
        "options": {
            "preserve_face": True,
            "handle_long_garment_delta": True
        }
    }
    response = client.post("/api/v1/tryon/jobs", json=payload)
    assert response.status_code == 202
    data = response.json()
    assert "job_id" in data
    assert data["status"] in ["queued", "processing", "completed"]
    assert data["store_id"] == "store_aurora_test"
    
    # Query job status
    job_id = data["job_id"]
    status_resp = client.get(f"/api/v1/tryon/jobs/{job_id}")
    assert status_resp.status_code == 200
    status_data = status_resp.json()
    assert status_data["job_id"] == job_id

