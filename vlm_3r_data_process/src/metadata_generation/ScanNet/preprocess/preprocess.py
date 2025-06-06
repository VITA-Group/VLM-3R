import argparse
import os
import subprocess
import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--num_workers', type=int, 
                       default=max(1, multiprocessing.cpu_count() // 2),
                       help='number of parallel workers')
    parser.add_argument('--force', action='store_true',
                       help='force reprocessing of already processed scenes')
    args = parser.parse_args()

    print(f"Preprocessing data from {args.input_dir} to {args.output_dir}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    scans_dir = os.path.join(args.input_dir, 'scans')
    scene_dirs = [d for d in os.listdir(scans_dir) if os.path.isdir(os.path.join(scans_dir, d))]
    
    # 过滤掉已处理的场景
    if not args.force:
        scene_dirs = filter_processed_scenes(scene_dirs, args.output_dir)
        print(f"Found {len(scene_dirs)} scenes to process after filtering")
    
    # 创建参数列表
    process_args = [(scans_dir, scene_dir, args.output_dir, args.force) for scene_dir in scene_dirs]
    
    # 使用进程池并行处理
    with Pool(processes=args.num_workers) as pool:
        list(tqdm(
            pool.imap(process_scene_wrapper, process_args),
            total=len(scene_dirs),
            desc="Preprocessing scenes"
        ))

def filter_processed_scenes(scene_dirs, output_dir):
    """过滤掉已经处理完成的场景"""
    unprocessed_scenes = []
    
    for scene_dir in scene_dirs:
        scene_output_dir = os.path.join(output_dir, scene_dir)
        
        # 检查是否已完成处理
        if not is_scene_processed(scene_output_dir):
            unprocessed_scenes.append(scene_dir)
    
    return unprocessed_scenes

def is_scene_processed(scene_output_dir):
    """检查场景是否已完成处理"""
    # 检查必要的目录和文件是否存在
    required_dirs = ['color', 'depth', 'pose']
    required_files = ['intrinsic/intrinsic_color.txt', 
                      'intrinsic/intrinsic_depth.txt', 
                      'intrinsic/extrinsic_color.txt', 
                      'intrinsic/extrinsic_depth.txt']
    
    # 检查目录
    for dir_name in required_dirs:
        dir_path = os.path.join(scene_output_dir, dir_name)
        if not os.path.isdir(dir_path):
            return False
        
        # 检查目录中是否有文件
        if len(os.listdir(dir_path)) == 0:
            return False
    
    # 检查文件
    for file_path in required_files:
        if not os.path.isfile(os.path.join(scene_output_dir, file_path)):
            return False
    
    # 检查最后一帧是否存在（确保处理完整）
    # 这需要知道总帧数，但我们可以检查是否有足够多的帧
    color_dir = os.path.join(scene_output_dir, 'color')
    depth_dir = os.path.join(scene_output_dir, 'depth')
    pose_dir = os.path.join(scene_output_dir, 'pose')
    
    # 确保三个目录中的文件数量相同
    color_count = len([f for f in os.listdir(color_dir) if f.endswith('.jpg')])
    depth_count = len([f for f in os.listdir(depth_dir) if f.endswith('.png')])
    pose_count = len([f for f in os.listdir(pose_dir) if f.endswith('.txt')])
    
    if color_count == 0 or depth_count == 0 or pose_count == 0:
        return False
    
    if color_count != depth_count or color_count != pose_count:
        return False
    
    return True

def process_scene_wrapper(args):
    """包装函数来处理多参数"""
    return preprocess_scene(*args)

def preprocess_scene(scans_dir, scene_dir, output_dir, force=False):    
    # Create output directory for current scene
    scene_output_dir = os.path.join(output_dir, scene_dir)
    
    # 如果不强制重新处理，检查是否已处理
    if not force and os.path.exists(scene_output_dir):
        if is_scene_processed(scene_output_dir):
            print(f"Scene {scene_dir} already processed, skipping...")
            return True
    
    os.makedirs(scene_output_dir, exist_ok=True)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    reader_path = os.path.join(current_dir, "reader.py")
    
    # 修改 OptimizedSensorData 类中的 num_workers 参数
    # 这需要在 reader.py 中添加对应的参数支持
    cmd = [
        "python", reader_path,
        "--filename", os.path.join(scans_dir, scene_dir, scene_dir + '.sens'),
        "--output_path", scene_output_dir,
        "--export_depth_images",
        "--export_color_images",
        "--export_poses",
        "--export_intrinsics",
        "--use_parallel",
        "--frame_skip", "10",
        "--image_size", "480", "640"
    ]

    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error processing scene {scene_dir}: {e}")
        return False

if __name__ == "__main__":
    main()
