# python process_videos.py --video_dir d:\development\autovideofacefinder\input --output_dir d:\development\autovideofacefinder\output --log_file processed_videos.log

import os
import shutil
import cv2
from facenet_pytorch import MTCNN
import argparse
from tqdm import tqdm
import torch

def process_video(video_path, output_dir, log_file):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mtcnn = MTCNN(keep_all=True, device=device)
    video = None
    frames_with_faces = 0  # Counter for frames with more than one face
    try:
        video = cv2.VideoCapture(video_path)
        if not video.isOpened():
            print(f"Error: Cannot open video file {video_path}")
            return

        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        step = int(max(1, frame_count / 45))  # Adjust step if necessary

        for frame_num in tqdm(range(0, frame_count, step), desc=f"Processing frames from {os.path.basename(video_path)}"):
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
            ret, frame = video.read()
            if not ret:
                break

            # Convert frame to RGB as MTCNN expects RGB but OpenCV provides BGR
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces, _ = mtcnn.detect(frame_rgb)

            if faces is not None and len(faces) > 1:
                frames_with_faces += 1
                if frames_with_faces > 20:
                    output_path = os.path.join(output_dir, os.path.basename(video_path))
                    shutil.copy(video_path, output_path)
                    with open(log_file, "a") as f:
                        f.write(f"{os.path.basename(video_path)}: {frames_with_faces} frames with multiple faces\n")
                    print(f"Detected multiple faces in over 50 frames of {video_path}. Video copied to output.")
                    break
    except Exception as e:
        print(f"An error occurred while processing {video_path}: {e}")
    finally:
        if video:
            video.release()

def process_videos(video_dir, output_dir, log_file):
    if not os.path.isdir(video_dir):
        print("Error: Video directory does not exist.")
        return
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    video_paths = [os.path.join(video_dir, f) for f in os.listdir(video_dir) if f.endswith((".mp4", ".avi"))]
    if not video_paths:
        print("No video files found in the input directory.")
        return

    print(f"Found {len(video_paths)} videos in the input directory.")
    
    for video_path in tqdm(video_paths, desc="Processing videos"):
        process_video(video_path, output_dir, log_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process videos to detect frames with multiple faces.")
    parser.add_argument("--video_dir", type=str, required=True, help="Path to the directory containing video files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to the output directory.")
    parser.add_argument("--log_file", type=str, required=True, help="Path to the log file where processed video names will be saved.")

    args = parser.parse_args()
    process_videos(args.video_dir, args.output_dir, args.log_file)
