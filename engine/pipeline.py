import time
from typing import Dict, Any, Optional, Callable
from PIL import Image

from engine.stages.pose import PoseExtractor
from engine.stages.segmenter import GarmentSegmenter
from engine.stages.pre_inpaint import DeltaGarmentInpainter
from engine.stages.vton import DiffusionTryOnEngine
from engine.stages.blender import IdentityAndDetailBlender

class VirtualTryOnPipeline:
    """
    Master 4-Stage Virtual Try-On Pipeline.
    
    Orchestrates:
      1. Stage 1: Grounded-SAM 2 & DensePose (Open-vocabulary garment parsing & 3D body pose)
      2. Stage 2: Delta Analysis & Pre-Inpainting (Solves Kurta-to-Shirt garment removal & pants inpainting)
      3. Stage 3: SOTA Diffusion Try-On (Leffa / IDM-VTON / CatVTON)
      4. Stage 4: Identity Preservation & Multi-Band Detail Blending (CodeFormer)
    """
    def __init__(self, model_preset: str = "leffa", device: str = "cpu"):
        self.device = device
        self.model_preset = model_preset
        
        # Initialize the 4 specialized stages
        self.pose_extractor = PoseExtractor(device=device)
        self.segmenter = GarmentSegmenter(device=device)
        self.delta_inpainter = DeltaGarmentInpainter(device=device)
        self.tryon_engine = DiffusionTryOnEngine(model_name=model_preset, device=device)
        self.blender = IdentityAndDetailBlender(device=device)

    def run(
        self,
        person_image: Image.Image,
        garment_image: Image.Image,
        garment_category: str = "upper_body",
        options: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[str, int], None]] = None
    ) -> Dict[str, Any]:
        """
        Executes end-to-end try-on generation with progress telemetry.
        """
        start_time = time.time()
        options = options or {}
        
        def report(stage: str, pct: int):
            if progress_callback:
                progress_callback(stage, pct)

        # -------------------------------------------------------------
        # STAGE 1: Body Pose & Open-Vocabulary Garment Parsing
        # -------------------------------------------------------------
        report("parsing", 15)
        pose_data = self.pose_extractor.extract_keypoints(person_image)
        densepose_map = self.pose_extractor.generate_densepose_map(person_image, pose_data)
        
        garment_info = self.segmenter.parse_person(person_image, pose_data)
        
        # -------------------------------------------------------------
        # STAGE 2: Garment Delta Analysis & Lower Body Pre-Inpainting
        # (The Kurta-to-Shirt algorithmic solution)
        # -------------------------------------------------------------
        report("pre_inpaint", 35)
        reconstruction_style = options.get("lower_wear_style", "auto_propagate")
        
        delta_result = self.delta_inpainter.process_delta(
            person_image=person_image,
            garment_info=garment_info,
            pose_data=pose_data,
            target_category=garment_category,
            reconstruction_style=reconstruction_style
        )
        
        reconstituted_canvas = delta_result["reconstituted_canvas"]
        agnostic_torso_mask = delta_result["agnostic_torso_mask"]
        delta_detected = delta_result["delta_detected"]
        
        # -------------------------------------------------------------
        # STAGE 3: SOTA Diffusion Try-On Synthesis
        # -------------------------------------------------------------
        report("diffusion", 65)
        raw_tryon = self.tryon_engine.generate_tryon(
            person_canvas=reconstituted_canvas,
            agnostic_mask=agnostic_torso_mask,
            garment_image=garment_image,
            densepose_map=densepose_map,
            options=options
        )
        
        # -------------------------------------------------------------
        # STAGE 4: Identity Preservation & Detail Blending
        # -------------------------------------------------------------
        report("blending", 90)
        preserve_face = options.get("preserve_face", True)
        final_image = self.blender.blend(
            original_person=person_image,
            tryon_result=raw_tryon,
            garment_info=garment_info,
            pose_data=pose_data,
            preserve_face=preserve_face
        )
        
        report("completed", 100)
        execution_time_ms = round((time.time() - start_time) * 1000, 2)
        
        return {
            "result_image": final_image,
            "execution_time_ms": execution_time_ms,
            "delta_detected": delta_detected,
            "detected_source_garment": garment_info.get("detected_type"),
            "diagnostics": {
                "y_bottom": garment_info.get("y_bottom"),
                "waist_y": pose_data.get("waist_y"),
                "knee_y": pose_data.get("knee_y"),
                "delta_mitigated": delta_detected
            }
        }

