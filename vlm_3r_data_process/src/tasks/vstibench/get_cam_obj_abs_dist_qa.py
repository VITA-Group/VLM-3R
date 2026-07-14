import os
import logging
import numpy as np
import open3d as o3d # Ensure open3d is imported

try:
    from ..base_qa_generator import BaseQAGenerator
    from ...utils.common_utils import generate_multiple_choice, sample_points_in_oriented_bbox_uniform # Import the utility
    from ...utils.camera_pose_utils import get_camera_position_from_c2w
except ImportError:
    # Fallback for running script directly or different structure
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from tasks.base_qa_generator import BaseQAGenerator
    from utils.common_utils import generate_multiple_choice, sample_points_in_oriented_bbox_uniform # Import the utility
    from utils.camera_pose_utils import get_camera_position_from_c2w

logger = logging.getLogger(__name__)

class AbsoluteDistanceQAGenerator(BaseQAGenerator):
    """
    Generates frame-level questions asking for the 5th percentile Euclidean distance
    (in meters) between the camera's position in a specific frame and the point cloud
    of a unique object instance (identified via scene metadata and PLY data),
    expecting a numerical answer.
    Uses preprocessed PLY files and scene metadata loaded via the base class.
    """
    def __init__(self):
        super().__init__()
        self.min_dist_threshold = 0.2 # Min distance for question validity
        self.max_dist_threshold = 10.0 # Max distance for question validity
        # scene_metadata is loaded by the base class as self.scene_annos
        # self.unique_instance_key = "object_instance_id" # Or a similar key from your metadata

    def get_default_args(self):
        """Return default arguments specific to frame-level absolute distance QA."""
        return {
            'question_template': 'VSTI_CAMERA_OBJ_DIST_TEMPLATE', # Updated template name
            'num_subsample': 6, # Max QAs per scene
            'question_type': 'camera_obj_abs_dist', # Updated type
            'output_filename_prefix': 'qa_camera_obj_abs_dist', # Updated prefix
        }

    def _get_camera_position(self, pose_matrix):
        """Extracts camera position (world coords) from T_c2w."""
        return get_camera_position_from_c2w(pose_matrix)

    def _create_o3d_bbox_from_meta(self, bbox_meta_dict):
        """
        Creates an open3d.geometry.OrientedBoundingBox from a metadata dictionary.
        bbox_meta_dict is expected to contain keys like 'centroid', 'normalizedAxes', and 'axesLengths'.
        """
        try:
            center = np.array(bbox_meta_dict["centroid"]).astype(np.float64).reshape(3, 1)
            # Consistent with common_utils.py:
            # R = np.array(ins["normalizedAxes"]).astype(np.float64).reshape(3, 3).T
            R = np.array(bbox_meta_dict["normalizedAxes"]).astype(np.float64).reshape(3, 3).T
            
            extent = np.array(bbox_meta_dict["axesLengths"]).astype(np.float64).reshape(3, 1)

            if not (center.shape == (3, 1) and R.shape == (3, 3) and extent.shape == (3, 1)):
                logger.error(
                    f"Shape mismatch for o3d.geometry.OrientedBoundingBox creation: "
                    f"center: {center.shape}, R: {R.shape}, extent: {extent.shape}"
                )
                return None
            return o3d.geometry.OrientedBoundingBox(center, R, extent)
        except KeyError as e:
            logger.error(f"Missing key in bbox_meta_dict for o3d bbox creation: {e}. Keys: {list(bbox_meta_dict.keys())}")
            return None
        except Exception as e:
            logger.error(f"Error creating o3d.geometry.OrientedBoundingBox: {e}", exc_info=True)
            return None

    def _get_all_instance_details_map(self, scene_name, scene_info):
        """
        Parses scene_info to create a map from instance_id to its details
        (category_name, bbox_meta). Performs validation on instance data.
        """
        all_instance_details = {}
        object_bboxes_dict = scene_info.get('object_bboxes', {})

        if not object_bboxes_dict:
            logger.warning(f"Scene {scene_name}: 'object_bboxes' not found or empty in scene_info. Cannot extract instance details.")
            return {}

        required_bbox_keys = ["centroid", "normalizedAxes", "axesLengths"]

        for category_name, instances_list in object_bboxes_dict.items():
            if not isinstance(instances_list, list):
                logger.warning(f"Scene {scene_name}: Expected a list of instances for category '{category_name}', but got {type(instances_list)}. Skipping category.")
                continue

            for obj_data in instances_list:
                if not isinstance(obj_data, dict):
                    logger.warning(f"Scene {scene_name}: Expected a dictionary for instance data under category '{category_name}', but got {type(obj_data)}. Skipping instance.")
                    continue

                instance_id = obj_data.get('instance_id')
                if instance_id is None:
                    logger.warning(f"Scene {scene_name}: Object under category '{category_name}' is missing 'instance_id'. Skipping.")
                    continue
            
                if not all(key in obj_data for key in required_bbox_keys):
                    missing_keys = [key for key in required_bbox_keys if key not in obj_data]
                    logger.warning(f"Scene {scene_name}: Object ID {instance_id} ('{category_name}') missing bbox keys: {missing_keys}. Has keys: {list(obj_data.keys())}. Skipping.")
                    continue

                try:
                    np.array(obj_data["centroid"]).astype(np.float64)
                    np.array(obj_data["normalizedAxes"]).astype(np.float64)
                    np.array(obj_data["axesLengths"]).astype(np.float64)
                except Exception as e:
                    logger.warning(f"Scene {scene_name}: Object ID {instance_id} ('{category_name}') has non-numeric or malformed bbox data. Error: {e}. Skipping.")
                    continue
                
                if instance_id in all_instance_details:
                     logger.warning(f"Scene {scene_name}: Duplicate instance_id {instance_id} encountered. Overwriting with data from category '{category_name}'.")

                all_instance_details[instance_id] = {
                    "category_name": category_name,
                    "bbox_meta": {key: obj_data[key] for key in required_bbox_keys}
                }
        
        if not all_instance_details:
            logger.info(f"Scene {scene_name}: No objects with valid instance IDs and required bbox metadata found in object_bboxes.")
            
        return all_instance_details

    def _calculate_potential_qas(self, scene_name, frames, all_instance_details_map):
        """
        Calculates distances for instances that are unique by category within each specific frame.
        Points are sampled from OrientedBoundingBox for these frame-unique instances.
        """
        potential_qas = []
        sampled_instance_points = {} # Memoization for sampled points across frames

        if not all_instance_details_map:
            logger.info(f"Scene {scene_name}: No instances with valid details map provided. Skipping distance calculation.")
            return []

        for frame_index, frame_data in enumerate(frames):
            frame_id = frame_data['frame_id']
            camera_pose_list = frame_data.get('camera_pose_camera_to_world')
            if not camera_pose_list:
                # logger.debug(f"Scene {scene_name}, Frame {frame_id}: Missing camera_pose_camera_to_world.")
                continue

            cam_pos_world = self._get_camera_position(camera_pose_list)
            if cam_pos_world is None:
                # logger.debug(f"Scene {scene_name}, Frame {frame_id}: Could not get camera position.")
                continue

            # --- Identify frame-specific unique instances by category (Simplified) ---
            visible_instance_details_this_frame = [] # Stores dicts: {"instance_id": id, "category_name": cat, "bbox_meta": meta}
            category_counts_this_frame = {}

            for bbox_info in frame_data.get('bboxes_2d', []):
                instance_id = bbox_info.get('instance_id')
                if instance_id is None: # Ensure instance_id exists and is not None
                    continue
                
                instance_master_details = all_instance_details_map.get(instance_id)
                if not instance_master_details: # Ensure instance_id is in the master map
                    # logger.warning(f"Scene {scene_name}, Frame {frame_id}: Instance ID {instance_id} from bboxes_2d not found in all_instance_details_map.")
                    continue
                
                category_name = instance_master_details["category_name"]
                
                visible_instance_details_this_frame.append({
                    "instance_id": instance_id,
                    "category_name": category_name,
                    "bbox_meta": instance_master_details["bbox_meta"]
                })
                category_counts_this_frame[category_name] = category_counts_this_frame.get(category_name, 0) + 1

            if not visible_instance_details_this_frame:
                # logger.debug(f"Scene {scene_name}, Frame {frame_id}: No valid instances with details found in bboxes_2d for this frame.")
                continue

            frame_specific_unique_instances = []
            for inst_details in visible_instance_details_this_frame:
                if category_counts_this_frame.get(inst_details["category_name"]) == 1:
                    frame_specific_unique_instances.append(inst_details)
            # --- End: Identify frame-specific unique instances ---

            if not frame_specific_unique_instances:
                # logger.debug(f"Scene {scene_name}, Frame {frame_id}: No frame-specific unique instances by category found.")
                continue

            # Iterate through these frame-specific unique instances for QA generation
            for instance_info in frame_specific_unique_instances:
                instance_id = instance_info["instance_id"]
                category_name = instance_info["category_name"]
                bbox_meta_dict = instance_info.get("bbox_meta") # Should exist based on construction

                # Sample points for this instance if not already sampled and cached
                if instance_id not in sampled_instance_points:
                    if bbox_meta_dict is None: # Should not happen
                        logger.warning(f"Scene {scene_name}, Frame {frame_id}, Instance {instance_id}: Missing 'bbox_meta' unexpectedly. Skipping.")
                        continue
                    
                    o3d_bbox = self._create_o3d_bbox_from_meta(bbox_meta_dict)
                    if o3d_bbox is None:
                        logger.warning(f"Scene {scene_name}, Frame {frame_id}, Instance {instance_id}: Could not create o3d_bbox. Skipping.")
                        continue

                    sampled_points_for_instance = sample_points_in_oriented_bbox_uniform(o3d_bbox)
                    if sampled_points_for_instance is None or sampled_points_for_instance.shape[0] == 0:
                        logger.warning(f"Scene {scene_name}, Frame {frame_id}, Instance {instance_id}: sample_points_in_oriented_bbox_uniform returned no points. BBox center: {o3d_bbox.center}, extent: {o3d_bbox.extent}. Skipping.")
                        continue
                    sampled_instance_points[instance_id] = sampled_points_for_instance
                
                # Check if points were successfully sampled and cached
                if instance_id not in sampled_instance_points:
                    # logger.debug(f"Scene {scene_name}, Frame {frame_id}, Instance {instance_id}: Points not available after sampling attempt. Skipping.")
                    continue

                object_points_world = sampled_instance_points[instance_id]

                distances = np.linalg.norm(object_points_world - cam_pos_world, axis=1)
                if distances.size == 0:
                    # logger.debug(f"Scene {scene_name}, Frame {frame_id}, Instance {instance_id}: No distances calculated.")
                    continue

                distance_min = np.min(distances)

                if self.min_dist_threshold <= distance_min <= self.max_dist_threshold:
                    potential_qas.append({
                        "frame_id": frame_id,
                        "frame_rank": frame_index,
                        "instance_id": instance_id,
                        "category_name": category_name,
                        "distance_min": round(distance_min, 1)
                    })
        return potential_qas

    def _format_qa_item(self, scene_name, scene_info, num_frames, qa_info):
        """
        Formats a single QA item dictionary based on the calculated info.

        Args:
            scene_name (str): The scene name.
            scene_info (dict): Scene metadata.
            num_frames (int): Total number of frames considered.
            qa_info (dict): Dictionary containing frame_id, category_name, distance_min.

        Returns:
            dict: The formatted QA item.
        """
        frame_id = qa_info["frame_id"]
        frame_rank = qa_info["frame_rank"] # Get the rank
        category_name = qa_info["category_name"]
        gt_distance = qa_info["distance_min"]

        question_text = self.question_template.format(
            object_name=category_name,
            frame_description=f"frame {frame_rank + 1} of {num_frames}" # Use rank (1-based)
        )

        # Generate multiple choice options
        options, mc_answer, self.answer_counts = generate_multiple_choice(
            gt_distance, 
            decimals=1, # Match the rounding of gt_distance
            answer_counts=self.answer_counts, 
            option_letters=self.option_letters
        )

        # Handle potential errors from generate_multiple_choice
        if mc_answer == "E":
             logger.warning(f"Scene {scene_name}, Frame {frame_id}, Instance {qa_info['instance_id']}: Could not generate valid multiple choice options for GT {gt_distance}. Skipping QA.")
             return None # Indicate failure to generate QA

        return {
            "dataset": self.args.dataset,
            "scene_name": scene_name,
            "video_path": scene_info.get("video_path"),
            "frame_indices": [frame_id], # Store 0-based index internally
            "question_type": self.args.question_type,
            "question": question_text,
            "options": options, # Use generated options
            "ground_truth": gt_distance,
            "mc_answer": mc_answer, # Use generated mc_answer
        }

    def generate_scene_qa(self, scene_name, scene_info, frame_info_for_scene):
        """Generates frame-level absolute 5th percentile distance QA pairs using bbox sampling from utility."""
        scene_qa_list = []

        # --- 1. Get All Instance Details from Scene Metadata ---
        all_instance_details_map = self._get_all_instance_details_map(scene_name, scene_info)
        if not all_instance_details_map:
            logger.info(f"Scene {scene_name}: No valid instance details map created from scene_info. Skipping QA generation.")
            return []

        # --- Get Original Total Frames First ---
        all_frames_in_metadata = frame_info_for_scene.get('frames', [])
        original_total_num_frames = len(all_frames_in_metadata)
        if original_total_num_frames == 0: # Handle case where 'frames' key is missing or empty list
            logger.info(f"Scene {scene_name}: No 'frames' key or empty frames list in frame_info_for_scene. Skipping QA generation.")
            return []

        # --- 2. Get Valid Frame Data (for QA logic) ---
        valid_frames = sorted([
            f for f in all_frames_in_metadata # Filter from the original list
            if f.get('camera_pose_camera_to_world')
        ], key=lambda x: x['frame_id'])

        if not valid_frames:
            logger.warning(f"Scene {scene_name}: No frames with camera poses found after filtering. Skipping QA generation.")
            return []
        # num_frames for question formatting will be original_total_num_frames

        # --- 3. Calculate Distances for Potential QAs (using frame-specific unique instances) ---
        potential_qas = self._calculate_potential_qas(
            scene_name, valid_frames, all_instance_details_map # Use valid_frames for calculation logic
        )

        # --- 4. Format Final QA Items ---
        for qa_info in potential_qas:
            # Pass original_total_num_frames for question formatting
            qa_item = self._format_qa_item(scene_name, scene_info, original_total_num_frames, qa_info)
            if qa_item:
                scene_qa_list.append(qa_item)

        if not scene_qa_list:
             logger.info(f"Scene {scene_name}: No valid QA pairs generated after distance filtering and bbox sampling via utility.")

        return scene_qa_list

if __name__ == '__main__':
    # Setup basic logging if running directly
    if not logging.getLogger().hasHandlers():
         logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    generator = AbsoluteDistanceQAGenerator()
    generator.run()
