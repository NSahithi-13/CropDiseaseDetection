# Crop Disease Detection using CNN

## Project Overview
This project is focused on detecting diseases in **crop leaves** using a **Convolutional Neural Network (CNN)**.  
Given an image of a leaf, the system predicts which disease (or healthy condition) it belongs to.  
The prediction is visualized along with the input image for better understanding.

The project uses the **PlantVillage dataset** for training and includes a pre-trained model for easy testing.

---

## Folder Structure

CropDiseaseDetection/

├── app/ # Optional for future UI/web app

├── dataset/ # PlantVillage dataset (not included, see instructions)
│ └── PlantVillage/
│ ├── Pepper__bell___Bacterial_spot/
│ ├── Pepper__bell___healthy/
│ ├── Potato___Early_blight/
│ └── ... (all classes)
├── scripts/ # Python scripts
│ ├── train_model.py # Train CNN model
│ └── predict.py # Predict disease on images
├── models/
│ └── crop_disease_model.h5 # Trained CNN model
├── test_images/ # Sample images for testing
├── notebooks/ # Optional Jupyter notebooks
├── README.md
├── requirements.txt # Required Python packages
├── venv/ # Virtual environment (ignored in git)
└── .gitignore # Files/folders ignored in git


---

## Dataset
We used the **[PlantVillage dataset](https://www.kaggle.com/datasets/emmarex/plantdisease)**.  

- The dataset contains **15 classes** of crops including Tomato, Potato, and Pepper.  
- Each class folder has multiple images of healthy and diseased leaves.  
- The dataset is **not included** in this repository due to its size.  

**Instructions to set up dataset locally:**
1. Download the dataset from [Kaggle link](https://www.kaggle.com/datasets/emmarex/plantdisease)  
2. Extract the files into:

## Technologies Used
- **Python** – Core programming language
- **TensorFlow & Keras** – For building and training CNN
- **OpenCV** – For image preprocessing and visual display
- **NumPy** – Array handling
- **Matplotlib** – Visualizations
- **Scikit-learn** – Train-test splitting & categorical encoding

## How It Works
1. **Data Preprocessing:** Images are loaded, resized to 64x64 pixels, normalized, and labeled.
2. **CNN Training:** A Convolutional Neural Network is trained on the PlantVillage dataset.
3. **Model Saving:** The trained model is saved as `crop_disease_model.h5`.
4. **Prediction:** 
   - Upload single or multiple leaf images.
   - The model predicts disease class with confidence.
   - Images and predictions are displayed visually using OpenCV.

## Installation

1. Clone the repo:
```bash
git clone https://github.com/NSahithi-13/CropDiseaseDetection.git
cd CropDiseaseDetection

2. Create a virtual environment:

python -m venv venv


3. Activate the environment:

# Windows
.\venv\Scripts\activate


4. Install required packages:

pip install -r requirements.txt

