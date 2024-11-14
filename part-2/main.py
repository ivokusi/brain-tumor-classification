import streamlit as st
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.preprocessing import image
import google.generativeai as genai
from tensorflow.keras.optimizers import Adamax
import plotly.graph_objects as go
from dotenv import load_dotenv
from openai import OpenAI
from PIL import Image
import numpy as np
import base64
import groq
import cv2
import os

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

openai_client = OpenAI()
groq_client = groq.Groq(api_key=GROQ_API_KEY)

output_dir = "saliency_maps"
os.makedirs(output_dir, exist_ok=True)

base_path = os.path.dirname(os.path.abspath(__file__))

# LLM Response

def encode_image(img_path):
  with open(img_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
  
def request_groq_model(prompt, img_path):

    base64_image = encode_image(img_path)

    chat_completion = groq_client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    { 
                        "type": "text", 
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        },
                    },
                ],
            }
        ],
        model="llama-3.2-11b-vision-preview",
    )

    return chat_completion.choices[0].message.content

def request_openai_model(prompt, img_path):

    base64_image = encode_image(img_path)

    chat_completion = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        }
                    },
                ],
            }
        ],
    )

    return chat_completion.choices[0].message

def request_gemini_model(prompt, img_path):

    img = Image.open(img_path)

    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    response = model.generate_content([prompt, img])

    return response.text

def generate_explanation(model, img_path, model_prediction, confidence):

    prompt = f"""
    As an expert neurologist, your task is to analyze and interpret a saliency map generated from a brain MRI scan.
    This saliency map was created by a deep learning model trained to classify brain tumors into one of four categories: glioma, meningioma, no tumor, or pituitary tumor.
    
    The model predicted that the tumor type in this image is '{model_prediction}' with a confidence of {confidence}%.

    In your response:
     - Identify the specific brain regions the model focused on, as shown by the highlighted areas in light cyan on the saliency map.
     - Use both the highlighted regions and the predicted tumor type to provide a well-rounded analysis, explaining why these regions may or may not support the model's prediction.
     - If the model’s focus areas appear misplaced or inconsistent with typical indicators of the predicted tumor type, discuss why this might suggest an inaccurate prediction.
     - Avoid mentioning overfitting or model confidence in your explanation. Focus solely on the MRI image and the areas highlighted by the saliency map.
     - Limit your explanation to 4 sentences to keep it concise.
    """

    if model == "groq":
        return request_groq_model(prompt, img_path)
    elif model == "openai":
        return request_openai_model(prompt, img_path)
    else:
        return request_gemini_model(prompt, img_path)

def generate_saliency_map(model_name, model, img, img_array, class_index, img_size):
    
    with tf.GradientTape() as tape:
        img_tensor = tf.convert_to_tensor(img_array)
        tape.watch(img_tensor)
        predictions = model(img_tensor)
        target_class = predictions[:, class_index]

    gradients = tape.gradient(target_class, img_tensor)
    gradients = tf.math.abs(gradients)
    gradients = tf.reduce_max(gradients, axis=-1)
    gradients = gradients.numpy().squeeze()

    # Resize gradients to match original image size
    gradients = cv2.resize(gradients, img_size)

    # Create a circular mask for the brain area
    center = (gradients.shape[0] // 2, gradients.shape[1] // 2)
    radius = min(center[0], center[1]) - 10
    y, x = np.ogrid[:gradients.shape[0], :gradients.shape[1]]
    mask = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius ** 2

    # Apply mask to gradients
    gradients = gradients * mask

    # Normalize only the brain area
    brain_gradients = gradients[mask]
    if brain_gradients.max() > brain_gradients.min():
        brain_gradients = (brain_gradients - brain_gradients.min()) / (brain_gradients.max() - brain_gradients.min())
    gradients[mask] = brain_gradients

    # Apply a higher threshold
    threshold = np.percentile(gradients[mask], 80)
    gradients[gradients < threshold] = 0

    # Apply more aggressive smoothing
    gradients = cv2.GaussianBlur(gradients, (11, 11), 0)

    # Create a heatmap overlay with enhanced contrast
    heatmap = cv2.applyColorMap(np.uint8(255 * gradients), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # Resize heatmap to match original image size
    heatmap = cv2.resize(heatmap, img_size)

    # Superimpose the heatmap on original image with increased opacity
    original_img = image.img_to_array(img)
    superimposed_img = heatmap * 0.7 + original_img * 0.3
    superimposed_img = superimposed_img.astype(np.uint8)

    img_path = os.path.join(output_dir, model_name, uploaded_file.name)
    os.makedirs(os.path.dirname(img_path), exist_ok=True)

    with open(img_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Save the saliency map
    cv2.imwrite(img_path, cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))

    return superimposed_img

# Load NN Model

def load_xception_model(model_path):

    img_shape = (299, 299, 3)

    base_model = tf.keras.applications.Xception(
        include_top=False,
        weights="imagenet",
        input_shape=img_shape,
        pooling="max"
    )

    model = Sequential([
        base_model,
        Flatten(),
        Dropout(rate=0.3),
        Dense(128, activation="relu"),
        Dropout(rate=0.25),
        Dense(4, activation="softmax")
    ])

    model.build((None,) + img_shape)

    model.compile(
        optimizer=Adamax(learning_rate=0.001), 
        loss="categorical_crossentropy", 
        metrics=["accuracy", Precision(), Recall()]
    )

    model.load_weights(model_path)

    return model

def load_resnet_model(model_path):

    img_shape = (299, 299, 3)

    base_model = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=img_shape,
        pooling="max"
    )

    model = Sequential([
        base_model,
        Flatten(),
        Dropout(rate=0.3),
        Dense(128, activation="relu"),
        Dropout(rate=0.25),
        Dense(4, activation="softmax")
    ])

    model.build((None,) + img_shape)

    model.compile(
        optimizer=Adamax(learning_rate=0.001), 
        loss="categorical_crossentropy", 
        metrics=["accuracy", Precision(), Recall()]
    )

    model.load_weights(model_path)

    return model

def run_model(model_name, model, uploaded_file, img_size):

    img = image.load_img(uploaded_file, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    labels = ["Glioma", "Meningioma", "No tumor", "Pituitary"]

    prediction = model.predict(img_array)

    class_index = np.argmax(prediction[0])
    result = labels[class_index]

    saliency_map = generate_saliency_map(model_name, model, img, img_array, class_index, img_size)

    return { 
        "name": model_name, 
        "prediction": result, 
        "confidence": round(prediction[0][class_index] * 100, 5),
        "saliency_map": saliency_map,
        "probability_distribution": prediction[0]
    }

def get_prediction_color(prediction):

    color_palette = {
        "No tumor": "#2E3D7C",
        "Meningioma": "#BA292E",
        "Glioma": "#E15D3A", 
        "Pituitary": "#FFA73C",
    }

    return color_palette.get(prediction, "#282528")

def draw_saliency_maps(models, uploaded_file):
    
    img = image.load_img(uploaded_file, target_size=(299, 299))

    cols = st.columns(len(models) + 1)

    with cols[0]:
        st.image(img, caption="Original Image", use_container_width=True)

    for idx, model in enumerate(models):

        with cols[idx + 1]:
            
            st.image(model["saliency_map"], caption=f"{model['name']}", use_container_width=True)

def draw_classification_results(models):

    labels = ["Glioma", "Meningioma", "No tumor", "Pituitary"]

    st.markdown(
        """
        <style>
        
        .model-card {
            background-color: #1e1e1e;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 10px;
            color: white;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.25);
        }
        
        .model-name {
            background-color: white;
            color: black;
            padding: 4px 8px;
            border-radius: 12px;
            font-weight: bold;
            display: inline-block;
            margin-bottom: 16px;
        }
        
        .label {
            font-size: 0.8em;
            color: #888;
            margin-right: 8px;
        }
        
        .prediction, .confidence {
            font-weight: bold;
            font-size: 0.9em;
        }
        
        .progress-bar-container {
            background-color: #333;
            height: 8px;
            border-radius: 5px;
            overflow: hidden;
            margin-top: 10px;
        }
        
        .progress-bar-fill {
            height: 100%;
            border-radius: 5px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(len(models))

    # Display each model's information in a card layout
    for idx, model in enumerate(models):
        
        with cols[idx]:

            sorted_indices = np.argsort(model['probability_distribution'])[::-1]

            label = labels[sorted_indices[0]]
            probability = model['probability_distribution'][sorted_indices[0]]

            probability_color = get_prediction_color(label)


            st.markdown(
                f"""
                <div class="model-card">
                    <div class="model-name">{model['name']}</div>
                        <div>
                        <span class="prediction" style="background-color: {probability_color}; padding: 2px 8px; border-radius: 8px; margin-right: 5px;">
                            {label}
                        </span>
                        <span class="prediction" style="background-color: {probability_color}; padding: 2px 8px; border-radius: 8px">
                            {round(probability * 100, 4)}%
                        </span>
                        <div class="progress-bar-container" style="width: 95%; display: inline-block; vertical-align: middle">
                            <div class="progress-bar-fill" style="width: {round(probability * 100, 4)}%; background-color: {probability_color};"></div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            with st.expander("Probability Distribution"):

                for index in sorted_indices[1:]:

                    label = labels[index]
                    probability = model['probability_distribution'][index]

                    probability_color = get_prediction_color(label)

                    st.markdown(
                        f"""
                        <div style="margin-bottom: 12px;">
                            <span class="prediction" style="background-color: {probability_color}; padding: 2px 8px; border-radius: 8px; margin-right: 5px;">
                                {label}
                            </span>
                            <span class="prediction" style="background-color: {probability_color}; padding: 2px 8px; border-radius: 8px">
                                {round(probability * 100, 4)}%
                            </span>
                            <br>
                            <div class="progress-bar-container" style="width: 95%; display: inline-block; vertical-align: middle">
                                <div class="progress-bar-fill" style="width: {round(probability * 100, 4)}%; background-color: {probability_color};"></div>
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

st.title("Brain Tumor Classification")
st.write("Uplaod an image of a brain MRI scan to classify.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    if st.session_state.get("uploaded_file") != uploaded_file:
        st.session_state.clear()
        st.session_state["uploaded_file"] = uploaded_file

    selected_models = st.multiselect(
        label="Select Models",
        options=["Xception", "ResNet50", "Custom CNN"],
        default=["Xception", "ResNet50", "Custom CNN"]
    )

    models = list()

    if "Xception" in selected_models and st.session_state.get("Xception") is None:
        model = load_xception_model(os.path.join(base_path, "..", "part-1", "models", "xception_model.weights.h5"))
        response = run_model("Xception", model, uploaded_file, (299, 299))
        st.session_state["Xception"] = response
        models.append(response)
    elif "Xception" in selected_models and st.session_state.get("Xception") is not None:
        models.append(st.session_state.get("Xception"))
    
    if "ResNet50" in selected_models and st.session_state.get("ResNet50") is None:
        model = load_resnet_model(os.path.join(base_path, "..", "part-1", "models", "resnet_model.weights.h5"))
        response = run_model("ResNet50", model, uploaded_file, (299, 299))
        st.session_state["ResNet50"] = response
        models.append(response)
    elif "ResNet50" in selected_models and st.session_state.get("ResNet50") is not None:
        models.append(st.session_state.get("ResNet50"))
    
    if "Custom CNN" in selected_models and st.session_state.get("Custom CNN") is None:
        model = load_model(os.path.join(base_path, "..", "part-1", "models", "cnn_model.keras"))
        response = run_model("Custom CNN", model, uploaded_file, (224, 224))
        st.session_state["Custom CNN"] = response
        models.append(response)
    elif "Custom CNN" in selected_models and st.session_state.get("Custom CNN") is not None:
        models.append(st.session_state.get("Custom CNN"))

    if selected_models:
        
            st.write("## Saliency Maps")

            draw_saliency_maps(models, uploaded_file)

            st.write("## Classification Results")

            draw_classification_results(models)

            llm_model = st.selectbox(
                "LLM Model",
                options=["Groq", "OpenAI", "Gemini"]
            )

            if llm_model is not None:
                
                st.write("## Explanation")

                for model in models:

                    saliency_map_path = os.path.join(output_dir, model["name"], uploaded_file.name)

                    if st.session_state.get(f"{model['name']}_{llm_model}_explanation") is None:
                        explanation = generate_explanation(llm_model, saliency_map_path, model["prediction"], model["confidence"])
                        st.session_state[f"{model['name']}_{llm_model}_explanation"] = explanation
                    
                    st.write(f"### {model['name']}")
                    st.write(st.session_state.get(f"{model['name']}_{llm_model}_explanation"))

                    with st.expander("Chat with LLM"):
                        
                        chat_container = st.container(height=300)

                        if st.session_state.get(f"{model['name']}_{llm_model}_history") is None:
                            st.session_state[f"{model['name']}_{llm_model}_history"] = list()

                        prompt = st.chat_input("Say something", key=f"{model['name']}_{llm_model}_chat") 

                        with chat_container:
                            
                            if prompt:
                                st.session_state[f"{model['name']}_{llm_model}_history"].append({"role": "user", "content": prompt})
                                st.session_state[f"{model['name']}_{llm_model}_history"].append({"role": "assistant", "content": f"Echo: {prompt}"})

                            for message in st.session_state[f"{model['name']}_{llm_model}_history"]:
                                st.chat_message(message["role"]).write(message["content"]) 
