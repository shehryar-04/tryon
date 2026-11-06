import numpy as np
from PIL import Image, ImageDraw
import pytest

from engine.stages.pose import PoseExtractor
from engine.stages.segmenter import GarmentSegmenter
from engine.stages.pre_inpaint import DeltaGarmentInpainter

def create_synthetic_kurta_person(width=600, height=900) -> Image.Image:
    """Creates a synthetic image of a person wearing a knee-length blue kurta."""
    img = Image.new("RGB", (width, height), (240, 240, 240))
    draw = ImageDraw.Draw(img)
    
    # Head & Face (y: 60 - 180)
    draw.ellipse([240, 60, 360, 180], fill=(210, 160, 120))
    
    # Blue Kurta (Shoulders at y=180 down to knees at y=640)
    # Note: Waist is at y=423, knees at y=648
    kurta_points = [
        (200, 180), (400, 180),
        (420, 640), (180, 640)
    ]
    draw.polygon(kurta_points, fill=(2, 132, 199)) # Blue
    
    # White Shalwar visible below knees (y: 640 - 820)
    draw.rectangle([210, 640, 285, 820], fill=(235, 235, 235))
    draw.rectangle([315, 640, 390, 820], fill=(235, 235, 235))
    
    return img

def test_kurta_to_shirt_delta_removal():
    """
    Validates that when a person is wearing a knee-length kurta and selects a shirt:
    1. The delta zone (lower half of kurta from waist to knees) is isolated.
    2. delta_detected is True.
    3. The agnostic torso mask strictly stops at the shirt hemline.
    4. Reconstituted canvas replaces the kurta flaps with restored trousers.
    """
    person_img = create_synthetic_kurta_person()
    w, h = person_img.size
    
    pose_extractor = PoseExtractor()
    pose_data = pose_extractor.extract_keypoints(person_img)
    
    segmenter = GarmentSegmenter()
    garment_info = segmenter.parse_person(person_img, pose_data)
    
    assert garment_info["is_long_garment"] is True, "Segmenter must detect long garment (kurta)"
    assert garment_info["y_bottom"] > pose_data["waist_y"], "Kurta bottom must be lower than waistline"
    
    inpainter = DeltaGarmentInpainter()
    result = inpainter.process_delta(
        person_image=person_img,
        garment_info=garment_info,
        pose_data=pose_data,
        target_category="upper_body",
        reconstruction_style="auto_propagate"
    )
    
    # Assertions
    assert result["delta_detected"] is True, "Must detect long-to-short garment mismatch"
    
    delta_mask = result["delta_mask"]
    agnostic_torso_mask = result["agnostic_torso_mask"]
    shirt_hem_y = result["shirt_hem_y"]
    
    # Verify delta mask only exists below the shirt hemline
    delta_y_indices, _ = np.where(delta_mask > 0)
    assert len(delta_y_indices) > 0, "Delta mask must not be empty"
    assert np.min(delta_y_indices) > shirt_hem_y, "Delta mask must only remove the kurta flaps below waist"
    
    # Verify agnostic torso mask only exists at or above shirt hemline
    torso_y_indices, _ = np.where(agnostic_torso_mask > 0)
    assert np.max(torso_y_indices) <= shirt_hem_y, "Torso mask must not extend into knees"
    
    # Verify the reconstituted canvas has eliminated the blue kurta color in the delta zone
    reconstituted_np = np.array(result["reconstituted_canvas"])
    
    # Sample a point in the delta zone (e.g. thigh area y=550, x=300)
    thigh_pixel = reconstituted_np[550, 300]
    # In the original image, thigh was blue (2, 132, 199)
    # In reconstituted image, it must be the inpainted trousers/shalwar tone, not the blue kurta
    is_blue_kurta = (thigh_pixel[0] == 2 and thigh_pixel[1] == 132 and thigh_pixel[2] == 199)
    assert not is_blue_kurta, "Blue kurta flaps in the delta region must be erased and replaced with trousers"
    
    print("\n[✓] Kurta-to-Shirt Delta Test Passed: Kurta flaps removed, trousers inpainted cleanly.")

if __name__ == "__main__":
    test_kurta_to_shirt_delta_removal()

