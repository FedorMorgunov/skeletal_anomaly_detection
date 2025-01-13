import os
from dataset import HRCrimeDataset
import config
import matplotlib.pyplot as plt

def load_video_list(file_path):
    with open(file_path, 'r') as file:
        videos = [line.strip() for line in file.readlines()]
    return videos

def plot_skeleton(skeleton):
    # Connections between joints for plotting
    connections = [
        (0, 1), (0, 2), (1, 3), (2, 4),
        (5, 7), (7, 9), (6, 8), (8, 10),
        (11, 13), (13, 15), (12, 14), (14, 16),
        (3, 5), (4, 6), (5, 6), (5, 11), (6, 12), (11, 12)
    ]
    
    x_coords = skeleton[::2]
    y_coords = skeleton[1::2]
    
    plt.figure()
    for connection in connections:
        x = [x_coords[connection[0]], x_coords[connection[1]]]
        y = [y_coords[connection[0]], y_coords[connection[1]]]
        plt.plot(x, y, 'bo-')
    plt.title('Skeleton Plot')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.gca().invert_yaxis() # Invert y-axis
    plt.show()


def get_all_videos(root_dir):
    trajectories_dir = os.path.join(root_dir, 'Trajectories')
    categories = [d for d in os.listdir(trajectories_dir) if os.path.isdir(os.path.join(trajectories_dir, d))]
    all_videos = []
    for category in categories:
        category_path = os.path.join(trajectories_dir, category)
        videos = [d for d in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, d))]
        for video_name in videos:
            all_videos.append(f"{category}/{video_name}")
    return all_videos

def main():
    # Load test video names
    test_videos = load_video_list(config.TEST_VIDEOS_FILE)
    
    # Get all videos and exclude test videos to get training videos
    all_videos = get_all_videos(config.ROOT_DIR)
    train_videos = list(set(all_videos) - set(test_videos))

    # Filter normal videos
    train_videos = [video for video in train_videos if 'Normal' in video]

    print(f"Total videos: {len(all_videos)}")
    print(f"Training videos: {len(train_videos)}")
    print(f"Testing videos: {len(test_videos)}")

    # Create datasets
    train_dataset = HRCrimeDataset(
        videos=train_videos,
        root_dir=config.ROOT_DIR,
        masks_dir=None,  # No frame-level labels for training data
        max_persons=config.MAX_PERSONS,
        sequence_length=config.SEQUENCE_LENGTH
    )

    test_dataset = HRCrimeDataset(
        videos=test_videos,
        root_dir=config.ROOT_DIR,
        masks_dir=config.MASKS_DIR,
        max_persons=config.MAX_PERSONS,
        sequence_length=config.SEQUENCE_LENGTH
    )

    print(f"Training dataset size (number of sequences): {len(train_dataset)}")
    print(f"Testing dataset size (number of sequences): {len(test_dataset)}")

    # Examine some samples from the training dataset
    print("\nSample from the training dataset:")
    sample_index = 0  # You can change this index to view different samples
    skeleton_sequence, label_sequence = train_dataset[sample_index]
    print(f"Skeleton sequence shape: {skeleton_sequence.shape}")
    print(f"Label sequence shape: {label_sequence.shape}")
    print(f"Label sequence: {label_sequence}")

    # Examine some samples from the testing dataset
    print("\nSample from the testing dataset:")
    skeleton_sequence_test, label_sequence_test = test_dataset[sample_index]
    print(f"Skeleton sequence shape: {skeleton_sequence_test.shape}")
    print(f"Label sequence shape: {label_sequence_test.shape}")
    print(f"Label sequence: {label_sequence_test}")
    # Plot the 1st skeleton in the 10th frame of the sample
    first_frame_skeleton = skeleton_sequence_test[9][:34].numpy()
    plot_skeleton(first_frame_skeleton)

if __name__ == '__main__':
    main()
