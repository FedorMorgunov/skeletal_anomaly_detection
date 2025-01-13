import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from dataset import HRCrimeDataset
from models.lstm_model import SkeletonAnomalyDetector
import config
import os
from tqdm import tqdm

def load_all_videos(root_dir):
    trajectories_dir = os.path.join(root_dir, 'Trajectories')
    categories = [d for d in os.listdir(trajectories_dir) if os.path.isdir(os.path.join(trajectories_dir, d))]
    all_videos = []
    for category in categories:
        category_path = os.path.join(trajectories_dir, category)
        if os.path.isdir(category_path):
            videos = os.listdir(category_path)
            for video_name in videos:
                video_path = os.path.join(category_path, video_name)
                if os.path.isdir(video_path):
                    all_videos.append(f"{category}/{video_name}")
    return all_videos

def main():
    # Get all videos
    all_videos = load_all_videos(config.ROOT_DIR)
    # Filter normal videos (Assuming categories with 'Normal' in name are normal)
    train_videos_normal = [video for video in all_videos if 'Normal' in video]
    # Use fewer videos (for quick debugging)
    train_videos_normal = train_videos_normal[:100]

    train_dataset = HRCrimeDataset(
        videos=train_videos_normal,
        root_dir=config.ROOT_DIR,
        max_persons=config.MAX_PERSONS,
        sequence_length=config.SEQUENCE_LENGTH
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )

    input_size = config.MAX_PERSONS * 34
    model = SkeletonAnomalyDetector(
        input_size=input_size,
        hidden_size=config.HIDDEN_SIZE,
        num_layers=config.NUM_LAYERS
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Since all labels are normal (0), we could train the model as a binary classifier
    # But here's the catch: Without anomalies, a standard BCE loss would push outputs towards 0.
    # One approach: Just train it to output low scores (near 0) for normal data.
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    for epoch in range(config.NUM_EPOCHS):
        model.train()
        running_loss = 0.0

        # Wrap your data loader with a tqdm for a progress bar
        for batch_idx, (skeleton_sequences, label_sequences) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
        ):
            skeleton_sequences = skeleton_sequences.to(device)
            labels = label_sequences[:, -1].to(device).float()

            optimizer.zero_grad()
            outputs = model(skeleton_sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{config.NUM_EPOCHS}] - Avg Loss: {avg_loss:.4f}")

    # Save the model if needed
    torch.save(model.state_dict(), "normal_only_model.pth")

if __name__ == '__main__':
    main()