import numpy as np
from PIL import Image, ImageDraw
import pytest

from engine.pipeline import VirtualTryOnPipeline
from tests.test_kurta_delta import create_synthetic_kurta_person

def create_synthetic_target_shirt(width=300, height=300) -> Image.Image:
    """Creates a synthetic target shirt image."""
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    # Red & Black plaid or stylish shirt
    draw.rectangle([80, 50, 220, 250], fill=(220, 38, 38))
    # Collar
    draw.polygon([(120, 50), (150, 90), (180, 50)], fill=(185, 28, 28))
    return img

def test_full_tryon_pipeline():
    """
    Executes the complete 4-stage Virtual Try-On pipeline.
    Validates progress reporting, output image integrity, and diagnostics.
    """
    person_img = create_synthetic_kurta_person()
    garment_img = create_synthetic_target_shirt()
    
    pipeline = VirtualTryOnPipeline(model_preset="leffa")
    
    progress_records = []
    def track_progress(stage: str, pct: int):
        progress_records.append((stage, pct))
        
    result = pipeline.run(
        person_image=person_img,
        garment_image=garment_img,
        garment_category="upper_body",
        options={
            "preserve_face": True,
            "handle_long_garment_delta": True,
            "lower_wear_style": "auto_propagate"
        },
        progress_callback=track_progress
    )
    
    # 1. Verify result structure
    assert "result_image" in result
    assert isinstance(result["result_image"], Image.Image)
    assert result["result_image"].size == person_img.size
    
    # 2. Verify progress telemetry
    stages = [p[0] for p in progress_records]
    assert "parsing" in stages
    assert "pre_inpaint" in stages
    assert "diffusion" in stages
    assert "blending" in stages
    assert "completed" in stages
    
    # 3. Verify Kurta Delta mitigation flag
    assert result["delta_detected"] is True, "Must report delta detected and mitigated"
    assert result["diagnostics"]["delta_mitigated"] is True
    
    print(f"\n[✓] Pipeline Test Passed in {result['execution_time_ms']}ms with full 4-stage telemetry.")

if __name__ == "__main__":
    test_full_tryon_pipeline()

