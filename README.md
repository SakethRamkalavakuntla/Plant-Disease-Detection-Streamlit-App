# 🌿 Plant Disease Detection using Deep Learning

This project focuses on detecting plant leaf diseases using deep learning. By analyzing images of infected plant leaves, the model identifies the specific disease affecting the crop. The aim is to provide an efficient tool for early diagnosis, helping farmers and agronomists take preventive actions to reduce crop loss.

---
## 🌾 Disease Categories

The model classifies images into one of the following **three plant disease categories**:

1. **Corn (Maize) - Common Rust**  
   - Caused by *Puccinia sorghi*, it manifests as orange to brown pustules on the leaf surface.
   - Can cause significant reduction in photosynthetic activity and crop yield.

2. **Potato - Early Blight**  
   - A fungal disease caused by *Alternaria solani*.
   - Presents as concentric dark spots on older leaves, leading to leaf drop and reduced tuber formation.

3. **Tomato - Bacterial Spot**  
   - Caused by *Xanthomonas campestris*.
   - Appears as small water-soaked lesions that become necrotic, often causing extensive leaf damage and fruit blemishes.

---
## 🧠 Model Overview

We developed and compared two models:

### 🔹 Custom CNN Model
- Built from scratch using Keras and TensorFlow.
- Performs well on simple datasets but showed limited generalization capabilities.

### 🔹 ResNet50 Model (Final Model)
- A **ResNet50** architecture pretrained on ImageNet.
- Fine-tuned on the plant disease dataset.
- Demonstrated superior accuracy, robustness, and generalization.
- Incorporates **data augmentation** techniques to improve performance on diverse real-world leaf images.

➡️ **ResNet50 was selected for deployment due to its better overall performance.**

---


## 💻 Streamlit Web Application

The trained ResNet model is deployed using a **Streamlit** web app with an intuitive interface for real-time predictions.

### Features:
- Upload a plant leaf image via the web UI.
- Instantly receive disease classification with a confidence score.
- Clear, minimalistic layout suitable for users without technical backgrounds.

### 📷 UI Previews:
- [Index Page](images/index_image.png)
- [Result Page](images/result_image.png)

---

## ☁️ AWS Deployment

The app is hosted on an **AWS EC2 instance**, making it publicly accessible.

### Deployment Highlights:
- Ubuntu-based EC2 instance with Python, Streamlit, and all required dependencies.
- App served with `streamlit run app.py --server.port 8501 --server.address 0.0.0.0`.
- Inbound rules configured to allow traffic on port 8501.
- Seamless access from any browser via the instance’s public IP or domain name.


---


To run this app locally:

```bash
git clone <your-repo-url>
cd <your-repo-folder>
pip install -r requirements.txt
streamlit run app.py


