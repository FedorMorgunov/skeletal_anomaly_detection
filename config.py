import os

# Paths
ROOT_DIR = os.path.dirname(os.path.realpath(__file__)) + '/HR-Crime_dataset'  # Replace with your dataset root directory
MASKS_DIR = ROOT_DIR + '/frame_level_masks'  # Replace with your masks directory
TEST_VIDEOS_FILE = ROOT_DIR + '/Anomaly_Test_HR.txt'  # Replace with the path to Anomaly_Test_HR.txt

# Dataset parameters
MAX_PERSONS = 5
SEQUENCE_LENGTH = 30

# Training parameters
BATCH_SIZE = 8
NUM_EPOCHS = 20
LEARNING_RATE = 0.001
HIDDEN_SIZE = 128
NUM_LAYERS = 1