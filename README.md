# Brain Tumor Detection and Diagnosis

## Overview

This project is divided into two main components: **Part 1** (Data Exploration and Model Training) and **Part 2** (Interactive Frontend and Model Deployment). The goal is to predict brain tumor types from MRI images and provide an interface for users to interact with the models and their outputs.

---

## Part 1: Data Exploration and Model Training

### Dataset
We used the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset), which contains labeled brain MRI images for tumor classification tasks.

### Models Trained
1. **Xception (Transfer Learning)**
2. **ResNet50 (Transfer Learning)**
3. **Custom CNN**

#### Key Features:
- **Transfer Learning:** Xception and ResNet50 models were fine-tuned using pretrained weights from the ImageNet dataset.
- **Custom Model Design:** A lightweight CNN tailored to this problem was developed for comparison.
- **Performance Guidance:** 
  - Model performance and training history were analyzed with the help of ChatGPT.
  - Suggestions were implemented to align the models with the task context and improve their accuracy and recall.

---

## Part 2: Interactive Frontend and Model Deployment

### Streamlit-Based Frontend
An intuitive user interface was built using the Streamlit library, allowing users to:
1. **Upload MRI Images:** Analyze brain MRI images using trained models.
2. **Tumor Prediction:** Identify tumor types and display predictions.
3. **Model Comparison:** Compare performance metrics and results across the different trained models.
4. **Saliency Maps:** Visualize areas of the MRI image that influenced the model’s prediction.

### AI-Powered Features
- **LLM Integration for Reports:**
  - Generate a detailed report about the tumor prediction and suggested next steps for the patient.
- **Empathetic Chatbot:**
  - Users can chat with a friendly, empathetic language model to ask questions about predictions or the generated report.

---

## Demo

Check out the youtube video of the project: [Brain Tumor Detection Demo](https://youtu.be/wOMQ4hLDZ5U).
Check out the live demo of the project: [Brain Tumor Detection Demo](https://customer-churn-prediction-101.streamlit.app/).
- **Example Images:** You can find example images in the [examples folder](examples).
