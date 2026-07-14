import os
import logging
import numpy as np
import random

try:
    from ..base_qa_generator import BaseQAGenerator
    from ...utils.common_utils import generate_multiple_choice # Import the function
    from ...utils.camera_pose_utils import get_camera_position_from_c2w
except ImportError:
    # Fallback for running script directly or different structure
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from tasks.base_qa_generator import BaseQAGenerator
    from utils.common_utils import generate_multiple_choice # Import the function
    from utils.camera_pose_utils import get_camera_position_from_c2w

logger = logging.getLogger(__name__)

class CameraDisplacementQAGenerator(BaseQAGenerator):
    """
    Generates sequence-level questions asking for the approximate Euclidean
    distance the camera traveled between two specified frames, expecting
    a numerical answer.
    """
    def __init__(self):
        super().__init__()
        self.min_frame_diff = 5 # Min frames between start and end
        self.max_frame_diff = 30 # Max frames between start and end
        self.min_displacement = 0.2 # Min required displacement in meters
        self.max_displacement = 10.0 # Max allowed displacement in meters
        # For multiple choice generation
        self.answer_counts = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
        self.option_letters = ['A', 'B', 'C', 'D']

    def get_default_args(self):
        """Return default arguments specific to camera displacement QA."""
        return {
            'question_template': 'VSDR_CAM_DISPLACE_TEMPLATE', # Assumes this exists
            'num_subsample': 6, # Max QAs per scene
            'question_type': 'camera_displacement',
            'output_filename_prefix': 'qa_camera_displacement',
        }

    def _get_camera_position(self, pose_matrix):
        """Extracts camera position (world coords) from T_c2w."""
        return get_camera_position_from_c2w(pose_matrix)

    def generate_scene_qa(self, scene_name, scene_info, frame_info_for_scene):
        """Generates camera displacement QA pairs."""
        scene_qa_list = []
        if not frame_info_for_scene or 'frames' not in frame_info_for_scene:
            logger.warning(f"Scene {scene_name}: Missing frame data. Skipping.")
            return []

        all_frames_in_metadata = frame_info_for_scene.get('frames', [])
        original_total_num_frames = len(all_frames_in_metadata)
        if original_total_num_frames == 0:
            logger.info(f"Scene {scene_name}: No frames key or empty frames list in frame_info_for_scene. Skipping QA generation.")
            return []

        valid_frames = sorted([f for f in all_frames_in_metadata if f.get('camera_pose_camera_to_world')],
                        key=lambda x: x['frame_id'])
        num_valid_frames = len(valid_frames) # Renamed for clarity, used for QA logic

        if num_valid_frames < self.min_frame_diff + 1: # Logic based on valid frames
            logger.debug(f"Scene {scene_name}: Needs at least {self.min_frame_diff + 1} valid frames with pose to satisfy min_frame_diff={self.min_frame_diff}. Found {num_valid_frames}. Skipping.")
            return []

        potential_qas = []
        processed_pairs = set()

        max_attempts_multiplier = 20
        absolute_max_attempts = 200
        
        max_total_attempts = max(self.args.num_subsample,
                                 min(self.args.num_subsample * max_attempts_multiplier, absolute_max_attempts))

        if num_valid_frames < 2: # Logic based on valid frames
            logger.debug(f"Scene {scene_name}: Needs at least 2 valid frames to pick a pair. Found {num_valid_frames}. Skipping.")
            return []

        for _ in range(max_total_attempts):
            if len(potential_qas) >= self.args.num_subsample:
                break

            idx1, idx2 = random.sample(range(num_valid_frames), 2) # Sample from valid frames
            
            i = min(idx1, idx2)
            j = max(idx1, idx2)

            if (i, j) in processed_pairs:
                continue
            processed_pairs.add((i, j))

            frame_diff = j - i
            if not (self.min_frame_diff <= frame_diff <= self.max_frame_diff):
                continue
            
            frame_start_data = valid_frames[i] # Use valid_frames
            frame_end_data = valid_frames[j]   # Use valid_frames

            pose_start_list = frame_start_data.get('camera_pose_camera_to_world')
            pose_end_list = frame_end_data.get('camera_pose_camera_to_world')

            if not pose_start_list or not pose_end_list:
                logger.warning(f"Scene {scene_name}, Frames {frame_start_data.get('frame_id', 'N/A')}-{frame_end_data.get('frame_id', 'N/A')} (indices {i},{j}): Missing pose data unexpectedly from valid_frames. Skipping pair.")
                continue

            pos_start_world = self._get_camera_position(pose_start_list)
            pos_end_world = self._get_camera_position(pose_end_list)

            if pos_start_world is None or pos_end_world is None:
                continue

            displacement = np.linalg.norm(pos_end_world - pos_start_world)

            if self.min_displacement <= displacement <= self.max_displacement:
                 potential_qas.append({
                      "frame_start_id": frame_start_data['frame_id'],
                      "frame_end_id": frame_end_data['frame_id'],
                      "start_rank": i, 
                      "end_rank": j,   
                      "displacement": round(displacement, 1)
                 })

        for qa_info in potential_qas:
            frame_start_id = qa_info["frame_start_id"]
            frame_end_id = qa_info["frame_end_id"]
            start_rank = qa_info["start_rank"] # This rank is within valid_frames
            end_rank = qa_info["end_rank"]     # This rank is within valid_frames
            gt_displacement = qa_info["displacement"]

            # Create descriptive frame strings using original_total_num_frames for context
            # The ranks i and j are based on the valid_frames list.
            # If you need frame ranks relative to original_total_num_frames, you'd need to map back frame_ids.
            # However, the question usually refers to "frame X of Y [valid/processed] frames".
            # For this change, we use original_total_num_frames as 'Y'.
            # The frame number (e.g., "frame {start_rank+1}") will still be based on the ordering within valid_frames.
            # This might require careful thought if the question implies original indexing.
            # Let's assume for now that "frame M of N" where N is original total is acceptable,
            # and M is the rank within the sub-selected valid frames.
            # A more robust way would be to use actual frame_id or find its original rank if needed by question template.
            # For now, the change is to use original_total_num_frames as the denominator.
            
            # If the question template implies ranks are from the original sequence, this needs more care.
            # Example: "What is the displacement from frame {original_start_rank+1} to frame {original_end_rank+1} of {original_total_num_frames}?"
            # Current: "frame {rank_in_valid_frames+1}"
            # Let's assume template uses ranks from the (potentially filtered) list being processed.
            
            start_frame_desc = f"frame {start_rank + 1}" # Rank within valid_frames
            end_frame_desc = f"frame {end_rank + 1} of {original_total_num_frames}" # Use original_total_num_frames for description

            question_text = self.question_template.format(
                start_frame_description=start_frame_desc,
                end_frame_description=end_frame_desc
            )

            # Generate multiple choice options
            options, mc_answer, self.answer_counts = generate_multiple_choice(
                gt_displacement,
                decimals=1, # Match the rounding of gt_displacement
                answer_counts=self.answer_counts,
                option_letters=self.option_letters
            )

            # Handle potential errors from generate_multiple_choice
            if mc_answer == "E":
                 logger.warning(f"Scene {scene_name}, Frames {frame_start_id}-{frame_end_id}: Could not generate valid multiple choice options for GT {gt_displacement}. Skipping QA.")
                 continue # Skip this QA item

            qa_item = {
                "dataset": self.args.dataset,
                "scene_name": scene_name,
                "video_path": scene_info.get("video_path") if scene_info else None, # scene_info might be None
                "frame_indices": [frame_start_id, frame_end_id],
                "question_type": self.args.question_type,
                "question": question_text,
                "options": options, # Use generated options
                "ground_truth": gt_displacement,
                "mc_answer": mc_answer, # Use generated mc_answer
            }
            scene_qa_list.append(qa_item)

        logger.debug(f"Scene {scene_name}: Generated {len(scene_qa_list)} camera displacement QA pairs.")
        return scene_qa_list

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s [%(levelname)s] - %(message)s')
    generator = CameraDisplacementQAGenerator()
    generator.run()
