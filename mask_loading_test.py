import numpy as np
import config

mask_path = config.MASKS_DIR + '/Arrest/Arrest001.npy'
frame_labels = np.load(mask_path)
print(frame_labels[::3]) # every 10th frame