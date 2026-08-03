categories = ['neutral', 'happy', 'surprise']

train_directory = 'train'
test_directory = 'test'

train_size = 5000
original_image_size = (48, 48)
image_size = (150, 150)
batch_size = 128
validation_split = 0.2

BOARD_SIZE = 3

# Stable path used by player.py and transfer/random models
basic_model_path = 'results/best_basic_model.keras'