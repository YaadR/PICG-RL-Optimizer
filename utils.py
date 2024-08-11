import glob
import os
from datetime import datetime
import cv2
import numpy as np
from tqdm import tqdm


def delete_all_files(directory):
    # Create a path pattern for all files in the directory
    files = glob.glob(os.path.join(directory, '*'))
    
    for file in files:
        try:
            os.remove(file)
            print(f"Deleted {file}")
        except Exception as e:
            print(f"Error deleting {file}: {e}")

def get_unique_filename(directory, filename):
    # Check if the file already exists
    base, extension = os.path.splitext(filename)
    counter = 1
    
    new_filename = filename
    while os.path.exists(os.path.join(directory, new_filename)):
        new_filename = f"{base}_{counter}{extension}"
        counter += 1

    return new_filename

def create_video_from_frames(frames_dir, output_video, frame_rate=20):

    # List all files in the directory
    directory, filename = os.path.split(output_video)
    valid_name = get_unique_filename(directory, filename)
    output_video = os.path.join(directory, valid_name)
    files = os.listdir(frames_dir)
    num_frames = len(files)  # Count the number of files
    
    # Read the first frame to get dimensions
    first_frame_path = os.path.join(frames_dir, 'frame_00000.png')
    frame = cv2.imread(first_frame_path)
    
    if frame is None:
        raise ValueError("The first frame could not be read. Check the frame path or format.")
    
    height, width, _ = frame.shape
    
    # Define video writer object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video, fourcc, frame_rate, (width, height))
    
    # Write each frame to the video
    for i in tqdm(range(num_frames),desc="Video Producing:"):
        frame_path = os.path.join(frames_dir, f'frame_{i:05}.png')
        frame = cv2.imread(frame_path)
        
        if frame is None:
            print(f"Warning: Frame {frame_path} could not be read. Skipping.")
            continue
        
        video_writer.write(frame)
    
    # Release video writer object
    video_writer.release()
    print(f"Video saved as {output_video}")
    delete_all_files(frames_dir)

def custom_argmax(p):
    max_p = np.zeros_like(p)  # Initialize an array of zeros with the same shape as p
    
    # Argmax for the first 3 elements
    first_max_idx = np.argmax(p[:3])
    max_p[first_max_idx] = 1
    
    # Argmax for the second 3 elements
    second_max_idx = np.argmax(p[3:]) + 3  # Add 3 to get the correct index in the original array
    max_p[second_max_idx] = 1
    
    return max_p

