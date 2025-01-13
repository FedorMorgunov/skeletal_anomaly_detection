import os

# Paths
ROOT_DIR = os.path.dirname(os.path.realpath(__file__)) + '/HR-Crime_dataset'  # Replace with your dataset root directory
MASKS_DIR = ROOT_DIR + '/frame_level_masks'  # Replace with your masks directory
TEST_VIDEOS_FILE = ROOT_DIR + '/Anomaly_Test_HR.txt'  # Replace with the path to Anomaly_Test_HR.txt

# Dataset parameters (for quick debugging)
MAX_PERSONS = 2
SEQUENCE_LENGTH = 30

# LSTM training parameters (for quick debugging)
BATCH_SIZE = 2
NUM_EPOCHS = 2
LEARNING_RATE = 0.001
HIDDEN_SIZE = 32
NUM_LAYERS = 1

# Transformer model training parameters (for quick debugging)
TRANSFORMER_D_MODEL = 32
TRANSFORMER_NHEAD = 2
TRANSFORMER_LAYERS = 1
TRANSFORMER_DROPOUT = 0.1