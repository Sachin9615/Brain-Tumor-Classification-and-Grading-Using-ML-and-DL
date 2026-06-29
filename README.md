Markdown
# 🧠 Brain Tumor Analysis System

This is a Python-based desktop application that automatically detects, classifies, and maps anomalies from brain MRI scans. The system provides an end-to-end pipeline from raw image upload to deep learning inference and visual diagnostics.

---

## 🚀 Key Features

* **4-Class Classification:** Identifies scans as **Glioma**, **Meningioma**, **Pituitary Tumor**, or **No Tumor**.
* **Lightweight Deployment:** Uses **TensorFlow Lite (TFLite)** to load a trained **ResNet50V2** model, allowing the app to run fast on standard laptops without requiring a heavy GPU.
* **Explainable AI (Saliency Map):** Implements a custom **Occlusion Saliency** loop that blocks out sections of the image to find exactly where the model is looking, generating a visual heatmap.
* **Automated Size & Location Tracking:** Calculates whether the tumor area is *Small, Medium, or Large* and determines its spatial location (e.g., *Superior — Left*) using NumPy array operations.
* **Responsive Layout:** Built with a dark-themed **Tkinter** interface that utilizes active window updates to prevent freezing during heavy calculations.

---

## 🛠️ Tech Stack

* **Language:** Python
* **Deep Learning:** TensorFlow Lite (`tf.lite`)
* **Image Processing:** OpenCV (`cv2`), Pillow (`PIL`)
* **Math Arrays:** NumPy
* **Data Visualization:** Matplotlib
* **GUI Framework:** Tkinter / TTK

---

## ⚙️ How It Works Under the Hood

1. **Preprocessing:** The uploaded MRI image is automatically resized to $224 \times 224$ pixels and normalized (scaled between `0.0` and `1.0`).
2. **Inference:** The processed image matrix is passed into the TFLite interpreter to extract classification probabilities.
3. **Occlusion Loop:** The app segments the image into a $14 \times 14$ grid and sequentially masks patches with black squares. Areas that cause the sharpest drop in prediction confidence are mapped out as the tumor zone.
4. **Output Generation:** Displays the metrics on the dashboard and opens a 3-panel Matplotlib figure showing the *Original MRI*, the *Saliency Heatmap*, and a *Blended Overlay View*.

---

## 💻 Installation & Setup

1. Clone this repository.
2. Install the required Python packages:
   ```bash
   pip install tensorflow numpy opencv-python pillow matplotlib
3. Run this project
    Bash
      python app.py
