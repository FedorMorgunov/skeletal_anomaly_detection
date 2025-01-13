import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import HRCrimeDataset
from models.transformer_model import SkeletonTransformer
import config

def load_all_videos(root_dir):
    """
    Example function to scan your dataset directory structure
    and return a list of 'Category/VideoName' for all videos.
    Modify or remove if you already have a custom approach.
    """
    trajectories_dir = os.path.join(root_dir, 'Trajectories')
    categories = [
        d for d in os.listdir(trajectories_dir)
        if os.path.isdir(os.path.join(trajectories_dir, d))
    ]
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
    # --------------------
    # 1) LOAD VIDEO LISTS
    # --------------------
    all_videos = load_all_videos(config.ROOT_DIR)

    # Filter only normal videos
    train_videos = [v for v in all_videos if "Normal" in v]
    # Use fewer videos (for quick debugging)
    train_videos = train_videos[:100]

    # --------------------
    # 2) CREATE DATASET & DATALOADER
    # --------------------
    train_dataset = HRCrimeDataset(
        videos=train_videos,
        root_dir=config.ROOT_DIR,
        max_persons=config.MAX_PERSONS,
        sequence_length=config.SEQUENCE_LENGTH
        # transform=..., if you have any data augmentation or normalization
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=2
    )

    # --------------------
    # 3) INSTANTIATE MODEL
    # --------------------
    input_size = config.MAX_PERSONS * 34
    model = SkeletonTransformer(
        input_size=input_size,
        d_model=config.TRANSFORMER_D_MODEL,
        nhead=config.TRANSFORMER_NHEAD,
        num_layers=config.TRANSFORMER_LAYERS,
        dropout=config.TRANSFORMER_DROPOUT
    )

    # --------------------
    # 4) PREPARE FOR TRAINING
    # --------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE)

    # --------------------
    # 5) TRAINING LOOP
    # --------------------
    print("Starting Training...")
    for epoch in range(config.NUM_EPOCHS):
        model.train()
        running_loss = 0.0

        for batch_idx, (skeleton_sequences, label_sequences) in enumerate(
            tqdm(train_loader, desc=f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
        ):
            skeleton_sequences = skeleton_sequences.to(device)  # shape: (batch_size, seq_len, input_size)
            # If all normal, label_sequences might be all zeros.
            labels = label_sequences[:, -1].to(device).float()  # shape: (batch_size,)

            optimizer.zero_grad()
            outputs = model(skeleton_sequences)  # shape: (batch_size,)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Compute average loss for the epoch
        avg_loss = running_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{config.NUM_EPOCHS}] - Average Loss: {avg_loss:.4f}")

    print("Training Completed.")

    # --------------------
    # 6) SAVE THE MODEL
    # --------------------
    save_path = "transformer_skeleton_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == '__main__':
    main()