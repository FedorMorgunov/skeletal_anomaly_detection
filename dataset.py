import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

class HRCrimeDataset(Dataset):
    def __init__(self, videos, root_dir, masks_dir=None, max_persons=5, sequence_length=10, transform=None):
        """
        Args:
            videos (list): List of video names in 'Category/VideoName' format.
            root_dir (str): Root directory of the dataset containing the 'Trajectories' folder.
            masks_dir (str, optional): Directory containing frame-level masks.
            max_persons (int): Maximum number of persons to consider per frame.
            sequence_length (int): Number of frames in each sequence.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.videos = videos
        self.root_dir = root_dir
        self.masks_dir = masks_dir
        self.max_persons = max_persons
        self.sequence_length = sequence_length
        self.transform = transform
        
        # Build a list of samples with video name and total frames
        self.samples = self._prepare_samples()
        
    def _prepare_samples(self):
        samples = []
        for video in self.videos:
            category, video_name = video.split('/')
            video_path = os.path.join(self.root_dir, 'Trajectories', category, video_name)
            if not os.path.isdir(video_path):
                continue  # Skip if video_path is not a directory
            frame_numbers = self._get_frame_numbers(video_path)
            if not frame_numbers:
                continue  # Skip if no frames found
            num_frames = max(frame_numbers)
            
            # Generate sequences
            for start_frame in range(1, num_frames - self.sequence_length + 2, 20):
                samples.append({
                    'category': category,
                    'video_name': video_name,
                    'start_frame': start_frame,
                    'end_frame': start_frame + self.sequence_length - 1
                })
        return samples

    def _get_frame_numbers(self, video_path):
        # Get all CSV files for persons
        if not os.path.exists(video_path):
            return []
        person_files = [f for f in os.listdir(video_path) if f.endswith('.csv')]
        frame_numbers = set()
        for person_file in person_files:
            csv_path = os.path.join(video_path, person_file)
            df = pd.read_csv(csv_path, header=None)
            frame_numbers.update(df[0].unique())
        return sorted(frame_numbers)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        category = sample_info['category']
        video_name = sample_info['video_name']
        start_frame = sample_info['start_frame']
        end_frame = sample_info['end_frame']
        
        # Load skeleton data and labels for the sequence
        skeleton_sequence = self._load_skeleton_sequence(category, video_name, start_frame, end_frame)
        label_sequence = self._load_label_sequence(category, video_name, start_frame, end_frame)
        
        # Apply any transformations
        if self.transform:
            skeleton_sequence = self.transform(skeleton_sequence)
        
        return skeleton_sequence, label_sequence
    
    def _load_skeleton_sequence(self, category, video_name, start_frame, end_frame):
        video_path = os.path.join(self.root_dir, 'Trajectories', category, video_name)
        person_files = [f for f in os.listdir(video_path) if f.endswith('.csv')]
        
        # Dictionary to hold dataframes for each person
        person_data = {}
        for person_file in person_files:
            person_id = os.path.splitext(person_file)[0]  # e.g., '0001'
            csv_path = os.path.join(video_path, person_file)
            df = pd.read_csv(csv_path, header=None)
            person_data[person_id] = df
        
        # List to hold skeleton data for each frame in the sequence
        sequence_data = []
        for frame_num in range(start_frame, end_frame + 1):
            frame_skeletons = self._get_frame_skeletons(person_data, frame_num)
            frame_input = self._process_skeletons(frame_skeletons)
            sequence_data.append(frame_input)
        
        # Stack to create a tensor of shape (sequence_length, input_size)
        skeleton_sequence = np.stack(sequence_data)
        skeleton_sequence = torch.tensor(skeleton_sequence, dtype=torch.float32)
        return skeleton_sequence
    
    def _get_frame_skeletons(self, person_data, frame_number):
        frame_skeletons = []
        for df in person_data.values():
            frame_df = df[df[0] == frame_number]
            if not frame_df.empty:
                # Extract skeleton coordinates (columns 1 to 34)
                skeleton = frame_df.iloc[0, 1:].values  # Shape: (34,)
                frame_skeletons.append(skeleton)
        return frame_skeletons  # List of skeletons in the frame
    
    def _process_skeletons(self, skeletons):
        # Normalize and pad/truncate skeletons to max_persons
        skeletons_processed = []
        for skeleton in skeletons:
            normalized_skeleton = self._normalize_skeleton(skeleton)
            skeletons_processed.append(normalized_skeleton)
        
        # Pad with zeros if fewer than max_persons
        while len(skeletons_processed) < self.max_persons:
            skeletons_processed.append(np.zeros(34))
        
        # Truncate if more than max_persons
        skeletons_processed = skeletons_processed[:self.max_persons]
        
        # Flatten the list of skeletons
        frame_input = np.concatenate(skeletons_processed)
        return frame_input  # Shape: (max_persons * 34,)
    
    def _normalize_skeleton(self, skeleton):
        # Assuming coordinates are within video frame dimensions
        # Normalize to [0, 1] range
        width, height = 320, 240  # Video dimensions 320 x 240
        normalized = skeleton.copy()
        normalized[::2] /= width   # Normalize x-coordinates
        normalized[1::2] /= height  # Normalize y-coordinates
        return normalized
    
    def _load_label_sequence(self, category, video_name, start_frame, end_frame):
        if self.masks_dir:
            # Load frame-level labels
            mask_path = os.path.join(self.masks_dir, category, f"{video_name}.npy")
            if os.path.exists(mask_path):
                frame_labels = np.load(mask_path)
                # Get labels for the sequence
                label_sequence = frame_labels[start_frame - 1:end_frame]
                label_sequence = torch.tensor(label_sequence, dtype=torch.long)
            else:
                print(f"Mask file not found: {mask_path}")
                # If mask file doesn't exist, default to normal labels
                label_sequence = torch.zeros(self.sequence_length, dtype=torch.long)
        else:
            # If masks are not available, default to normal (0)
            label_sequence = torch.zeros(self.sequence_length, dtype=torch.long)
        return label_sequence