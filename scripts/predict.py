import tensorflow as tf
import numpy as np
import cv2
import os

# --------- CONFIG ---------
IMAGE_SIZE = 64        # model input size
DISPLAY_SIZE = 150     # displayed image size
MODEL_PATH = "models/crop_disease_model.h5"
FOLDER_PATH = "test_image"
MAX_VISIBLE_ROWS = 3   # number of image rows visible at once
PADDING = 15

# Class names
class_names = [
    'Pepper__bell___Bacterial_spot',
    'Pepper__bell___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_Septoria_leaf_spot',
    'Tomato_Spider_mites_Two_spotted_spider_mite',
    'Tomato__Target_Spot',
    'Tomato__Tomato_YellowLeaf__Curl_Virus',
    'Tomato__Tomato_mosaic_virus',
    'Tomato_healthy'
]

# --------- LOAD MODEL ---------
model = tf.keras.models.load_model(MODEL_PATH)

# --------- PROCESS IMAGES ---------
annotated_images = []
for img_file in os.listdir(FOLDER_PATH):
    img_path = os.path.join(FOLDER_PATH, img_file)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Cannot read image: {img_file}")
        continue

    # Prepare for model
    img_input = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE)) / 255.0
    img_input = np.expand_dims(img_input, axis=0)

    # Predict
    predictions = model.predict(img_input, verbose=0)
    class_index = np.argmax(predictions)
    confidence = predictions[0][class_index] * 100
    predicted_label = class_names[class_index]

    # Annotate image
    display_img = cv2.resize(img, (DISPLAY_SIZE, DISPLAY_SIZE))
    text = f"{predicted_label} ({confidence:.1f}%)"
    (w, h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(display_img, (0,0), (w+6,h+6), (255,255,255), -1)
    cv2.putText(display_img, text, (3,h+3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1, cv2.LINE_AA)

    annotated_images.append(display_img)

# --------- SCROLLABLE GRID ---------
cols = 5  # images per row
rows = (len(annotated_images) + cols - 1) // cols
img_h, img_w, c = DISPLAY_SIZE, DISPLAY_SIZE, 3
scroll_height = MAX_VISIBLE_ROWS * (img_h + PADDING) + PADDING
grid_width = cols * (img_w + PADDING) + PADDING
grid_full = np.ones((rows * (img_h + PADDING) + PADDING, grid_width, 3), dtype=np.uint8) * 200

# Fill full grid
for idx, img in enumerate(annotated_images):
    r = idx // cols
    c_idx = idx % cols
    y = r * (img_h + PADDING) + PADDING
    x = c_idx * (img_w + PADDING) + PADDING
    grid_full[y:y+img_h, x:x+img_w] = img

# Scrollable window
scroll_pos = 0
while True:
    view = grid_full[scroll_pos:scroll_pos+scroll_height, :, :]
    cv2.imshow("Crop Disease Predictions", view)
    key = cv2.waitKey(50) & 0xFF
    if key == 27:  # ESC to exit
        break
    elif key == ord('w'):  # scroll up
        scroll_pos = max(0, scroll_pos - (img_h + PADDING))
    elif key == ord('s'):  # scroll down
        scroll_pos = min(grid_full.shape[0] - scroll_height, scroll_pos + (img_h + PADDING))

cv2.destroyAllWindows()
