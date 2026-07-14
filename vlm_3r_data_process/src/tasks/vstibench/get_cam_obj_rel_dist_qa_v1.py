import os
import logging
import numpy as np
import random
import open3d as o3d

try:
    # Use BaseQAGenerator consistent with abs dist if possible, check base class details
    # Assuming BaseQAGenerator handles common loading/saving logic
    from ..base_qa_generator import BaseQAGenerator
    from ...utils.common_utils import from_options_to_mc_answer, sample_points_in_oriented_bbox_uniform
    from ...utils.camera_pose_utils import get_camera_position_from_c2w
except ImportError:
    # Fallback for running script directly or different structure
    import sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    # Adjust base class import if needed
    from tasks.base_qa_generator import BaseQAGenerator
    from utils.common_utils import from_options_to_mc_answer, sample_points_in_oriented_bbox_uniform
    from utils.camera_pose_utils import get_camera_position_from_c2w

logger = logging.getLogger(__name__)

class RelativeDistanceQAGeneratorV1(BaseQAGenerator):
    """
    Generates frame-level multiple-choice questions (4 options) asking which object,
    among unique object instances PRESENT IN THE CURRENT FRAME,
    is physically closest to the camera in a specific frame,
    based on the 5th percentile distance to the object's *sampled* point cloud from its BBox.
    This is the V1 version with 4 choices.
    """
    def __init__(self):
        super().__init__()
        self.min_dist_threshold = 0.5 # Minimum distance for an object to be considered
        self.ambiguity_threshold = 0.15 # Min distance difference between options
        # --- Sampling Configuration ---
        self.use_sampling = True # Set to True to enable point sampling from BBox

        self.num_choices = 4
        
    def get_default_args(self):
        """Return default arguments specific to frame-level relative distance QA (V1 - 4 choices)."""
        return {
            'question_template': f'VSTI_CAMERA_OBJ_REL_DIST_TEMPLATE_V1',
            'num_subsample': 6,
            'question_type': f'camera_obj_rel_dist_v1',
            'output_filename_prefix': f'qa_camera_obj_rel_dist_v1',
            'num_choices': 4
        }

    def _get_camera_position(self, pose_matrix):
        """Extracts camera position (world coords) from T_c2w."""
        return get_camera_position_from_c2w(pose_matrix)

    def _format_qa_item(self, scene_name, scene_info, frame_id, frame_rank, num_frames, potential_options):
        """
        Selects options, formats the QA item if valid.
        Uses self.num_choices (hardcoded to 4 for V1).
        """
        current_num_choices = self.num_choices # Directly use the hardcoded value

        if len(potential_options) < current_num_choices:
            # logger.debug(f"Scene {scene_name}, Frame {frame_id}: Need >= {current_num_choices} valid objects. Found {len(potential_options)}. Skipping.")
            return None

        potential_options.sort(key=lambda x: x['distance'])
        closest_object = potential_options[0]
        gt_category = closest_object["category"]
        gt_distance = closest_object["distance"]

        num_distractors_needed = current_num_choices - 1
        options_candidates = [obj for obj in potential_options[1:] if obj["category"] != gt_category]
        
        if len(options_candidates) < num_distractors_needed:
            # logger.debug(f"Scene {scene_name}, Frame {frame_id}: Not enough distinct category candidates for {num_distractors_needed} distractors. Skipping QA.")
            return None

        final_options_cats = [gt_category]
        final_options_dists = [gt_distance]
        random.shuffle(options_candidates)

        for cand in options_candidates:
            is_ambiguous = False
            for existing_dist in final_options_dists:
                if abs(cand["distance"] - existing_dist) < self.ambiguity_threshold:
                    is_ambiguous = True
                    break
            if not is_ambiguous and cand["category"] not in final_options_cats:
                final_options_cats.append(cand["category"])
                final_options_dists.append(cand["distance"])
            if len(final_options_cats) == current_num_choices:
                break

        if len(final_options_cats) < current_num_choices:
            # logger.debug(f"Scene {scene_name}, Frame {frame_id}: Could not find {current_num_choices} non-ambiguous, distinct options. Skipping.")
            return None

        options_out, mc_answer, self.answer_counts = from_options_to_mc_answer(
            final_options_cats, gt_category, self.answer_counts, self.option_letters
        )

        if not mc_answer or mc_answer == "E":
            logger.warning(f"Scene {scene_name}, Frame {frame_id}: Could not generate valid MC answer. Options: {final_options_cats}, GT: {gt_category}")
            return None

        format_args = {
            "frame_description": f"frame {frame_rank + 1} of {num_frames}",
        }
        # Template should expect choice_a, choice_b, choice_c, choice_d for V1
        for i in range(len(options_out)): 
            format_args[f"choice_{self.option_letters[i].lower()}"] = options_out[i]
        
        question_text = self.question_template.format(**format_args)
        formatted_options = [f"{self.option_letters[idx]}. {opt}" for idx, opt in enumerate(options_out)]

        qa_item = {
            "dataset": self.args.dataset,
            "scene_name": scene_name,
            "video_path": scene_info.get("video_path"),
            "frame_indices": [frame_id],
            "question_type": self.args.question_type,
            "question": question_text,
            "options": formatted_options,
            "ground_truth": gt_category,
            "mc_answer": mc_answer,
        }
        return qa_item

    # --- Helper methods copied/adapted from AbsoluteDistanceQAGenerator ---
    def _create_o3d_bbox_from_meta(self, bbox_meta_dict):
        """
        Creates an open3d.geometry.OrientedBoundingBox from a metadata dictionary.
        bbox_meta_dict is expected to contain keys like 'centroid', 'normalizedAxes', and 'axesLengths'.
        """
        try:
            center = np.array(bbox_meta_dict["centroid"]).astype(np.float64).reshape(3, 1)
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
    # --- End of helper methods ---

    def generate_scene_qa(self, scene_name, scene_info, frame_info_for_scene):
        """Generates QA using unique instances PRESENT IN EACH FRAME as candidates, sampling points from their BBoxes."""
        scene_qa_list = []
        
        # 1. Get All Instance Details (including BBox) from Scene Metadata
        if not scene_info:
            logger.warning(f"Scene {scene_name}: Missing scene_info. Skipping.")
            return []
        all_instance_details_map = self._get_all_instance_details_map(scene_name, scene_info)
        if not all_instance_details_map:
            logger.info(f"Scene {scene_name}: No valid instance details (with bboxes) found in scene_info. Skipping QA generation.")
            return []

        if not frame_info_for_scene or 'frames' not in frame_info_for_scene:
            logger.warning(f"Scene {scene_name}: Missing frame data or 'frames' key. Skipping.")
            return []

        all_frames_in_metadata = frame_info_for_scene.get('frames', [])
        original_total_num_frames = len(all_frames_in_metadata)
        if original_total_num_frames == 0:
            logger.info(f"Scene {scene_name}: No frames key or empty frames list in frame_info_for_scene. Skipping QA generation.")
            return []

        valid_frames = sorted([
            f for f in all_frames_in_metadata # Filter from the original list
            if f.get('camera_pose_camera_to_world')
        ], key=lambda x: x['frame_id'])

        if not valid_frames:
            logger.warning(f"Scene {scene_name}: No frames with camera pose found after filtering. Skipping.")
            return []
        # num_frames for formatting will be original_total_num_frames (passed to _format_qa_item)
        # num_valid_frames = len(valid_frames) # This is the count of frames to iterate over for QA logic

        sampled_obb_points_cache = {} # Cache for sampled points from BBoxes, keyed by instance_id
        
        logger_prefix_sampling = f"Scene {scene_name}: "
        if self.use_sampling:
            logger.info(f"{logger_prefix_sampling}Using BBox point sampling for relative distance (V1).")
        else:
            logger.info(f"{logger_prefix_sampling}BBox point sampling disabled. Instances will be skipped (V1).")

        for frame_rank, frame_data in enumerate(valid_frames): # Iterate over valid_frames
            frame_id = frame_data['frame_id']
            # camera_pose_list is already guaranteed by the valid_frames filter
            camera_pose_list = frame_data.get('camera_pose_camera_to_world') 
            cam_pos_world = self._get_camera_position(camera_pose_list)
            if cam_pos_world is None:
                # This might occur if _get_camera_position has further internal checks failing
                logger.warning(f"Scene {scene_name}, Frame {frame_id} (Rank {frame_rank}): Could not extract camera position from a supposedly valid frame. Skipping frame.")
                continue

            # --- Identify frame-specific unique instances (unique by category) ---
            # This logic needs to be within the loop for each frame.
            visible_instance_details_this_frame = [] 
            category_counts_this_frame = {}
            
            # Instances visible in this frame based on bboxes_2d
            visible_instances_bboxes_2d = frame_data.get('bboxes_2d', [])
            if not visible_instances_bboxes_2d:
                # logger.debug(f"Scene {scene_name}, Frame {frame_id}: No bboxes_2d. Skipping frame for rel dist.")
                continue

            for bbox_info in visible_instances_bboxes_2d:
                instance_id_from_bbox = bbox_info.get("instance_id")
                if instance_id_from_bbox is None: # Ensure instance_id exists and is not None
                    continue
                
                instance_master_details = all_instance_details_map.get(instance_id_from_bbox)
                if not instance_master_details: # Ensure instance_id is in the master map
                    # logger.warning(f"Scene {scene_name}, Frame {frame_id}: Instance ID {instance_id_from_bbox} from bboxes_2d not found in all_instance_details_map.")
                    continue
                
                category_name_from_master = instance_master_details["category_name"]
                
                visible_instance_details_this_frame.append({
                    "instance_id": instance_id_from_bbox,
                    "category_name": category_name_from_master,
                    "bbox_meta": instance_master_details["bbox_meta"] # Get bbox_meta from master details
                })
                category_counts_this_frame[category_name_from_master] = category_counts_this_frame.get(category_name_from_master, 0) + 1
            
            if not visible_instance_details_this_frame:
                # logger.debug(f"Scene {scene_name}, Frame {frame_id}: No valid instances with details found in bboxes_2d for this frame.")
                continue

            frame_specific_unique_instances_for_options = []
            for inst_details in visible_instance_details_this_frame:
                if category_counts_this_frame.get(inst_details["category_name"]) == 1: # Unique by category for this frame
                    frame_specific_unique_instances_for_options.append(inst_details)
            
            if not frame_specific_unique_instances_for_options or len(frame_specific_unique_instances_for_options) < self.num_choices:
                # logger.debug(f"Scene {scene_name}, Frame {frame_id}: Not enough unique instances ({len(frame_specific_unique_instances_for_options)}) for {self.num_choices} choices. Skipping frame.")
                continue
            # --- End: Identify frame-specific unique instances ---

            # Calculate distances for these unique instances
            frame_potential_options = [] # Store {"category": cat_name, "distance": dist, "instance_id": id}
            for instance_info in frame_specific_unique_instances_for_options:
                instance_id = instance_info["instance_id"]
                category_name = instance_info["category_name"]
                bbox_meta = instance_info["bbox_meta"] # This should exist
                object_points_world = None

                if self.use_sampling:
                    if instance_id in sampled_obb_points_cache:
                        object_points_world = sampled_obb_points_cache[instance_id]
                    else:
                        o3d_bbox = self._create_o3d_bbox_from_meta(bbox_meta)
                        if o3d_bbox:
                            # Use the utility for sampling
                            points = sample_points_in_oriented_bbox_uniform(o3d_bbox)
                            if points is not None and points.shape[0] > 0:
                                sampled_obb_points_cache[instance_id] = points
                                object_points_world = points
                            else:
                                logger.warning(f"Scene {scene_name}, Frame {frame_id}, Instance {instance_id} ('{category_name}'): BBox sampling returned no points. Skipping instance for this frame.")
                                continue # Skip this instance for this frame
                        else:
                            logger.warning(f"Scene {scene_name}, Frame {frame_id}, Instance {instance_id} ('{category_name}'): Could not create o3d_bbox. Skipping instance for this frame.")
                            continue # Skip this instance for this frame
                
                if object_points_world is None: # If sampling disabled or failed
                    # logger.info(f"Scene {scene_name}, Frame {frame_id}, Instance {instance_id}: No points available (sampling disabled or failed). Skipping instance for distance calc.")
                    continue

                distances_to_points = np.linalg.norm(object_points_world - cam_pos_world, axis=1)
                if distances_to_points.size == 0:
                    # logger.debug(f"Scene {scene_name}, Frame {frame_id}, Instance {instance_id}: No distances calculated (no points?).")
                    continue
                
                min_dist_to_obj = np.min(distances_to_points)

                if min_dist_to_obj >= self.min_dist_threshold:
                    frame_potential_options.append({
                        "category": category_name,
                        "distance": round(min_dist_to_obj, 2), # Round distance
                        "instance_id": instance_id
                    })
            
            # Try to format a QA item for this frame if enough options
            if frame_potential_options and len(frame_potential_options) >= self.num_choices:
                # Pass original_total_num_frames for formatting
                qa_item = self._format_qa_item(scene_name, scene_info, frame_id, frame_rank, original_total_num_frames, frame_potential_options)
                if qa_item:
                    scene_qa_list.append(qa_item)

        # Subsample if more QAs are generated than requested
        if self.args.num_subsample > 0 and len(scene_qa_list) > self.args.num_subsample:
            logger.info(f"Scene {scene_name}: Generated {len(scene_qa_list)} relative distance QAs (V1), subsampling to {self.args.num_subsample}.")
            scene_qa_list = random.sample(scene_qa_list, self.args.num_subsample)
        
        logger.info(f"Scene {scene_name}: Successfully generated {len(scene_qa_list)} relative distance QA items (V1).")
        return scene_qa_list

if __name__ == '__main__':
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s [%(levelname)s] - %(message)s')
    
    logger.info("Generating QA for V1 (4 choices)")
    generator = RelativeDistanceQAGeneratorV1()
    generator.run()
