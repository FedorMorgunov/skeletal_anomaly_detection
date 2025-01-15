from dataset import AvenueDataset
import matplotlib.pyplot as plt

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
    
    for (j1, j2) in connections:
        x1, y1 = x_coords[j1], y_coords[j1]
        x2, y2 = x_coords[j2], y_coords[j2]

        # Skip plotting if either joint is still (0,0):
        if (x1 == 0 and y1 == 0) or (x2 == 0 and y2 == 0):
            continue
        
        # Otherwise draw the line/segment
        plt.plot([x1, x2], [y1, y2], 'ro-')

    plt.gca().invert_yaxis()
    plt.title("Skeleton Plot (Skipping Zero Joints)")
    plt.show()

def main():
    # Create the training dataset
    train_dataset = AvenueDataset(
        root_dir='/Users/fedor/Desktop/skeletal fedor/Avenue',   # e.g., '/data/HRCrime'
        train=True,                    # True for training set
        max_persons=2,
        sequence_length=30,
        step=20,                       # Frame stride
        transform=None                 # Any custom transforms if needed
    )

    # Create the testing dataset
    test_dataset = AvenueDataset(
        root_dir='/Users/fedor/Desktop/skeletal fedor/Avenue',   # same root directory
        train=False,                   # False for testing set
        max_persons=2,
        sequence_length=30,
        step=20,
        transform=None
    )

    print(f"Training dataset size (number of sequences): {len(train_dataset)}")
    print(f"Testing dataset size (number of sequences): {len(test_dataset)}")

    # Examine some samples from the training dataset
    print("\nSample from the training dataset:")
    sample_index = 40  # You can change this index to view different samples
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
    first_frame_skeleton = skeleton_sequence_test[10][:34].numpy()
    plot_skeleton(first_frame_skeleton)

if __name__ == '__main__':
    main()
