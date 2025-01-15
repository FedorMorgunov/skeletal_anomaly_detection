import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from data import compute_bounding_box
from interpolation import fill_missing_joints_linear

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
        """
        Example: replace naive normalization with bounding-box–based normalization
        using `compute_bounding_box`.
        """
        # Convert the skeleton from shape (34,) to (17,2):
        coords_2d = skeleton.reshape(-1, 2)  # shape: (17, 2)

        # 1) Compute bounding box for the current skeleton
        width, height = 320, 240  # or your actual resolution
        left, right, top, bottom = compute_bounding_box(coords_2d.flatten(), 
                                                        video_resolution=[width, height],
                                                        return_discrete_values=False)
        
        # 2) For example, center skeleton on bounding-box "top-left"
        if (right - left) > 0 and (bottom - top) > 0:
            coords_2d[:, 0] = (coords_2d[:, 0] - left) / (right - left)
            coords_2d[:, 1] = (coords_2d[:, 1] - top) / (bottom - top)
        else:
            # Fallback if bounding box is invalid
            coords_2d[:] = 0.0

        # 3) Return as flattened array again
        return coords_2d.flatten()
    
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

class AvenueDataset(Dataset):
    def __init__(self, root_dir, train=True, max_persons=5, sequence_length=10, step=20,
                 transform=None):
        self.root_dir = root_dir
        self.train = train
        self.max_persons = max_persons
        self.sequence_length = sequence_length
        self.step = step
        self.transform = transform
        
        if self.train:
            self.data_dir = os.path.join(self.root_dir, 'training', 'trajectories')
            self.mask_dir = None
        else:
            self.data_dir = os.path.join(self.root_dir, 'testing', 'trajectories')
            self.mask_dir = os.path.join(self.root_dir, 'testing', 'test_frame_mask')
        
        self.samples = self._prepare_samples()

    def _prepare_samples(self):
        samples = []
        for folder_name in os.listdir(self.data_dir):
            folder_path = os.path.join(self.data_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue
            frame_numbers = self._get_frame_numbers(folder_path)
            if not frame_numbers:
                continue
            num_frames = max(frame_numbers)
            for start_frame in range(0, num_frames - self.sequence_length + 1, self.step):
                samples.append({
                    'folder_name': folder_name,
                    'start_frame': start_frame,
                    'end_frame': start_frame + self.sequence_length - 1
                })
        return samples

    def _get_frame_numbers(self, folder_path):
        frame_numbers = set()
        csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        for csv_file in csv_files:
            csv_path = os.path.join(folder_path, csv_file)
            df = pd.read_csv(csv_path, header=None)
            frame_numbers.update(df[0].unique())
        return sorted(frame_numbers)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        folder_name = sample_info['folder_name']
        start_frame = sample_info['start_frame']
        end_frame = sample_info['end_frame']
        
        skeleton_sequence = self._load_skeleton_sequence(folder_name, start_frame, end_frame)
        label_sequence = self._load_label_sequence(folder_name, start_frame, end_frame)
        
        if self.transform:
            skeleton_sequence = self.transform(skeleton_sequence)
        
        return skeleton_sequence, label_sequence
    
    def _load_skeleton_sequence(self, folder_name, start_frame, end_frame):
        """
        Reads CSVs in `folder_name`, retrieves frames [start_frame..end_frame], 
        pads/truncates them to `max_persons`, then does on-the-fly interpolation of missing joints.
        """
        folder_path = os.path.join(self.data_dir, folder_name)
        csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]
        
        dataframes = {}
        for csv_file in csv_files:
            path = os.path.join(folder_path, csv_file)
            df = pd.read_csv(path, header=None)
            dataframes[csv_file] = df
        
        sequence_data = []
        for frame_num in range(start_frame, end_frame + 1):
            frame_skeletons = self._get_frame_skeletons(dataframes, frame_num)
            frame_input = self._process_skeletons(frame_skeletons)  
            sequence_data.append(frame_input)
        
        # Now shape is (T, max_persons * 34)
        skeleton_sequence = np.stack(sequence_data, axis=0)  # shape: (sequence_length, max_persons*34)

        ## (1) Reshape to (T, 2*J) so interpolation sees each pair (x,y). 
        ##     If max_persons=2, then you have 68 coords => treat them as 34 "joints" in 2D:
        skeleton_sequence = self._interp_reshape_and_fill(skeleton_sequence)

        # Convert to torch
        skeleton_sequence = torch.tensor(skeleton_sequence, dtype=torch.float32)
        return skeleton_sequence
    
    def _get_frame_skeletons(self, dataframes, frame_number):
        frame_skeletons = []
        for df in dataframes.values():
            frame_df = df[df[0] == frame_number]
            if not frame_df.empty:
                # columns 1..34 => 17 joints
                skeleton = frame_df.iloc[0, 1:].values
                frame_skeletons.append(skeleton)
        return frame_skeletons

    def _process_skeletons(self, skeletons):
        """
        Normalizes each skeleton to (width=640, height=360), 
        then pads/truncates to self.max_persons, then flattens.
        """
        processed = []
        for skel in skeletons:
            # Example normalization
            skel = self._normalize_skeleton(skel)
            processed.append(skel)
        
        while len(processed) < self.max_persons:
            processed.append(np.zeros(34))
        processed = processed[:self.max_persons]
        
        return np.concatenate(processed, axis=0)  # shape (max_persons * 34)

    def _normalize_skeleton(self, skeleton):
        """
        Example: replace naive normalization with bounding-box–based normalization
        using `compute_bounding_box`.
        """
        # Convert the skeleton from shape (34,) to (17,2):
        coords_2d = skeleton.reshape(-1, 2)  # shape: (17, 2)

        # 1) Compute bounding box for the current skeleton
        width, height = 640, 360  # or your actual resolution
        left, right, top, bottom = compute_bounding_box(coords_2d.flatten(), 
                                                        video_resolution=[width, height],
                                                        return_discrete_values=False)
        
        # 2) For example, center skeleton on bounding-box "top-left"
        if (right - left) > 0 and (bottom - top) > 0:
            coords_2d[:, 0] = (coords_2d[:, 0] - left) / (right - left)
            coords_2d[:, 1] = (coords_2d[:, 1] - top) / (bottom - top)

            # 3) Clip to [0,1] to avoid negative or >1 coords
            coords_2d[:, 0] = np.clip(coords_2d[:, 0], 0.0, 1.0)
            coords_2d[:, 1] = np.clip(coords_2d[:, 1], 0.0, 1.0)
        else:
            # Fallback if bounding box is invalid
            coords_2d[:] = 0.0

        # 3) Return as flattened array again
        return coords_2d.flatten()

    def _load_label_sequence(self, folder_name, start_frame, end_frame):
        if self.train:
            return torch.zeros(self.sequence_length, dtype=torch.long)
        else:
            mask_file = f"{folder_name}.npy"
            mask_path = os.path.join(self.mask_dir, mask_file)
            if os.path.exists(mask_path):
                frame_labels = np.load(mask_path)
                label_seq = frame_labels[start_frame:end_frame+1]
                return torch.tensor(label_seq, dtype=torch.long)
            else:
                return torch.zeros(self.sequence_length, dtype=torch.long)
    
    def _interp_reshape_and_fill(self, skeleton_sequence):
        """
        Example function that reshapes (T, max_persons*34) => (T, 2*J) for interpolation,
        calls fill_missing_joints_linear, then reshapes back if needed.
        """
        # If max_persons=2 => we have 68 coords => treat them as 34 "joints" in 2D
        # If max_persons=1 => we have 34 coords => treat them as 17 "joints"
        T, total_coords = skeleton_sequence.shape

        # We'll just pass it to the interpolation as (T, total_coords). 
        # That means "2 * (#joints) = total_coords", so #joints = total_coords/2
        # We'll interpret zeros as missing x,y pairs:
        # Make sure it is float64 or float32
        skeleton_sequence = skeleton_sequence.astype(np.float64)

        # Interpolate
        filled = fill_missing_joints_linear(skeleton_sequence)

        # returned shape is still (T, total_coords)
        return filled


def fill_missing_joints_linear(skeleton_sequence):
    """
    Simple interpolation for missing joints coded as (x=0, y=0). 
    skeleton_sequence shape: (T, 2*J).
    """
    filled = skeleton_sequence.copy()
    num_frames, xy_dims = filled.shape
    num_joints = xy_dims // 2

    for j in range(num_joints):
        x_vals = filled[:, 2*j]
        y_vals = filled[:, 2*j + 1]
        missing_mask = (x_vals == 0) & (y_vals == 0)

        # If all missing, skip
        if missing_mask.all():
            continue

        valid_idx = np.where(~missing_mask)[0]
        invalid_idx = np.where(missing_mask)[0]
        # If the first or last frames are missing, clamp
        first_valid, last_valid = valid_idx[0], valid_idx[-1]
        for idx in range(0, first_valid):
            x_vals[idx] = x_vals[first_valid]
            y_vals[idx] = y_vals[first_valid]
        for idx in range(last_valid+1, num_frames):
            x_vals[idx] = x_vals[last_valid]
            y_vals[idx] = y_vals[last_valid]

        # Recompute after clamping
        missing_mask = (x_vals == 0) & (y_vals == 0)
        valid_idx = np.where(~missing_mask)[0]
        invalid_idx = np.where(missing_mask)[0]

        if len(valid_idx) > 1 and len(invalid_idx) > 0:
            x_vals[missing_mask] = np.interp(invalid_idx, valid_idx, x_vals[valid_idx])
            y_vals[missing_mask] = np.interp(invalid_idx, valid_idx, y_vals[valid_idx])
        
        filled[:, 2*j] = x_vals
        filled[:, 2*j + 1] = y_vals
    
    return filled
