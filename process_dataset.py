import cv2
import numpy as np
import os
import mediapipe as mp
from data_collection import mediapipe_detection_new, extract_keypoints_from_results, DATA_PATH, sequence_length
from db_models import get_session, Gloss, Video

# Ensure base dataset path exists
os.makedirs(DATA_PATH, exist_ok=True)

def process_video(video_path, output_dir):
    """
    Reads a video, extracts exactly `sequence_length` frames uniformly spaced,
    and runs Mediapipe extraction on them.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return False

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if frame_count < sequence_length:
        print(f"Warning: Video {video_path} has only {frame_count} frames, expected at least {sequence_length}.")
        cap.release()
        return False
        
    # Calculate uniform frame indices to sample exactly sequence_length frames
    indices = np.linspace(0, frame_count - 1, sequence_length, dtype=int)
    
    for frame_num, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            print(f"Failed to read frame {idx} from {video_path}")
            # Pad with zeros if we fail to read a frame
            keypoints = np.zeros(258) 
        else:
            image, pose_result, hand_result = mediapipe_detection_new(frame)
            keypoints = extract_keypoints_from_results(pose_result, hand_result)
            
        npy_path = os.path.join(output_dir, str(frame_num))
        np.save(npy_path, keypoints)
            
    cap.release()
    return True

def process_dataset():
    """
    Iterates through downloaded videos in the SQL database
    And exports mediapipe landmarks to MP_Data.
    """
    session = get_session()
    
    # Get all glosses
    glosses = session.query(Gloss).all()
    
    for gloss in glosses:
        # Get downloaded videos for this gloss that haven't been processed yet
        videos = session.query(Video).filter_by(
            gloss_id=gloss.id,
            status='DOWNLOADED',
            processed=False
        ).all()
        
        if not videos:
            continue
            
        print(f"Processing action: {gloss.word}")
        
        # We need a sequence number for each video of this action
        # Let's count how many sequences we already have in MP_Data for this action
        action_path = os.path.join(DATA_PATH, gloss.word)
        os.makedirs(action_path, exist_ok=True)
        
        existing_sequences = len(os.listdir(action_path))
        
        for sequence_offset, video in enumerate(videos):
            sequence_num = existing_sequences + sequence_offset
            output_dir = os.path.join(action_path, str(sequence_num))
            os.makedirs(output_dir, exist_ok=True)
            
            video_file = video.downloaded_path
            if not video_file or not os.path.exists(video_file):
                print(f"  -> File not found: {video_file}")
                continue
                
            print(f"  -> Extracting features from {os.path.basename(video_file)} into sequence {sequence_num}...")
            success = process_video(video_file, output_dir)
            if not success:
                print(f"  -> Skipping {video_file} due to errors.")
            else:
                # Mark as processed in database
                video.processed = True
                session.commit()
                
    print("\nDataset processing via SQL complete. You can now run train.py")

if __name__ == '__main__':
    process_dataset()
