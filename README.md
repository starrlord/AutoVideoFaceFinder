# Video Face Detection

This Python script processes video files in a given directory and detects frames with multiple faces using the MTCNN (Multi-task Cascaded Convolutional Networks) face detection model from the `facenet_pytorch` library. Videos with more than a specified number of frames containing multiple faces are copied to an output directory, and their names are logged in a file.

## Requirements

- Python 3.x
- PyTorch
- torchvision
- torchaudio
- CUDA (for GPU acceleration)

## Installation

1. Install PyTorch, torchvision, and torchaudio with CUDA support:

```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

2. Install the required dependencies:

```
pip install facenet_pytorch opencv-python tqdm
```

## Usage

1. Clone the repository or download the script file.

2. Open a terminal and navigate to the directory containing the script.

3. Run the script with the following command:

```
python process_videos_torch_parallel.py --video_dir /path/to/video/directory --output_dir /path/to/output/directory --log_file /path/to/log/file.txt
```

Replace `/path/to/video/directory` with the path to the directory containing the video files you want to process, `/path/to/output/directory` with the path to the directory where you want to save the videos with multiple faces, and `/path/to/log/file.txt` with the path to the log file where you want to save the names of the processed videos.

4. The script will process the videos in the specified directory and save the videos with multiple faces in the output directory. The names of the processed videos will be logged in the specified log file.

## Code Overview

- The script uses the MTCNN face detection model from the `facenet_pytorch` library to detect faces in video frames.
- It processes the videos in parallel using multiprocessing to speed up the execution.
- For each video, it samples frames at a specified interval (default is 45 frames) and checks if there are more than one face in each frame.
- If a video has more than a specified number of frames (default is 20) with multiple faces, it is copied to the output directory, and its name is logged in the log file.
- The script uses OpenCV for reading video frames and tqdm for displaying progress bars.

## Notes

- Ensure that you have sufficient disk space in the output directory to store the copied videos.
- The script assumes that the video files have the extensions `.mp4` or `.avi`. Modify the script if your video files have different extensions.
- Adjust the `step` variable in the `process_video` function to change the number of frames sampled from each video. Increasing the step will make the script faster but may miss some frames with multiple faces.
- Modify the threshold for the number of frames with multiple faces (default is 20) in the `process_video` function based on your requirements.