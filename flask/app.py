from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import Adamax
from flask import Flask, request, jsonify
import google.generativeai as genai
from openai import OpenAI
import tensorflow as tf
from groq import Groq
import numpy as np
import base64
import cv2
import os
import io

flask_app = Flask(__name__)

# NN Models (helper functions)

def generate_saliency_map(model_name, model, img, img_size, img_array, class_index, file):

    base_path = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(base_path, "images", model_name)
    os.makedirs(output_dir, exist_ok=True)
    
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

    img_path = os.path.join(output_dir, file.filename)

    with open(img_path, "wb") as f:
        f.write(file.read())

    # Save the saliency map
    cv2.imwrite(img_path, cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))

    return superimposed_img

def run_model(model_name, model, img_size, file):
    # Read the file into a BytesIO object
    img_bytes = file.read()  # Read the file content
    img = image.load_img(io.BytesIO(img_bytes), target_size=img_size)  # Load image from BytesIO
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    labels = ["Glioma", "Meningioma", "No tumor", "Pituitary"]

    prediction = model.predict(img_array)

    class_index = np.argmax(prediction[0])
    result = labels[class_index]

    saliency_map = generate_saliency_map(model_name, model, img, img_size, img_array, class_index, file)

    return { 
        "name": model_name, 
        "prediction": result, 
        "confidence": round(prediction[0][class_index] * 100, 5),
        "saliency_map": saliency_map.tolist(),
        "probability_distribution": prediction[0].tolist()
    }

# NN Models (main functions)

@flask_app.route("/run_xception", methods=["POST"])
def run_xception():

    base_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_path, "models", "xception_model.weights.h5")

    file = request.files["file"]

    img_size = (299, 299)
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

    return jsonify(run_model("Xception", model, img_size, file)), 200

@flask_app.route("/run_resnet", methods=["POST"])
def run_resnet():

    base_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_path, "models", "resnet_model.weights.h5")

    file = request.files["file"]

    img_size = (299, 299)
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

    return jsonify(run_model("ResNet50", model, img_size, file)), 200

@flask_app.route("/run_cnn", methods=["POST"])
def run_cnn():

    base_path = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_path, "models", "cnn_model.keras")

    file = request.files["file"]

    img_size = (224, 224)

    model = load_model(model_path)

    return jsonify(run_model("CustomCNN", model, img_size, file)), 200

# LLM Models (helper functions)

def encode_image(img_path):
  with open(img_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')
  
def request_groq_model(prompt, base64_image):

    groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

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

def request_openai_model(prompt, base64_image):

    openai_client = OpenAI()
    
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

    return chat_completion.choices[0].message.content

def request_gemini_model(prompt, base64_image):

    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

    data = {
        "role": "user",
        "parts": [
            {"text": prompt},
            {
                "inlineData": {
                    "data": base64_image,
                    "mimeType": "image/jpeg"
                }
            }
        ]
    }

    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    response = model.generate_content([data])

    return response.text

# LLM Models (main functions)

@flask_app.route('/generate-explanation', methods=['POST'])
def generate_explanation():

    base_path = os.path.dirname(os.path.abspath(__file__))

    data = request.json

    nn_model = data["nn_model"]
    llm_model = data["llm_model"]
    prediction = data["prediction"]
    confidence = data["confidence"]

    image_path = os.path.join(base_path, "images", nn_model, data["file_name"])
    base64_image = encode_image(image_path)

    prompt = f"""
    As an expert neurologist, your task is to analyze and interpret a saliency map generated from a brain MRI scan. This saliency map was created by a deep learning model trained to classify brain tumors into one of four categories: glioma, meningioma, no tumor, or pituitary tumor.
    
    The model predicted that the tumor type in this image is '{prediction}' with a confidence of {confidence}%.
    
    Please provide a comprehensive report that includes the following:

    1. **Model's Prediction Analysis:** Identify the specific brain regions the model focused on, as shown by the highlighted areas in light cyan on the saliency map. Explain how these regions either support or contradict the predicted tumor type. Avoid discussing overfitting or confidence levels.

    2. **Additional Insights:** Describe potential implications of the model's prediction based on typical signs of each tumor type. Discuss any specific features that may be atypical, noteworthy, or require further examination.

    3. **Relevant Historical Cases:** Reference any similar cases in neurological oncology that could provide context for this prediction, noting any patterns, outcomes, or diagnostic pathways that may assist in understanding this patient's case.

    4. **Next Steps for Patient and Doctors:** Outline recommended next steps for further diagnosis or treatment, taking into account the model's prediction and observed regions in the MRI. Include any additional tests, consultations, or specific treatment plans that would ensure comprehensive care.

    **Keep your response concise but thorough, using a maximum of 6 sentences per section.**
    """


    if llm_model == "Groq":
        return jsonify({ "response": request_groq_model(prompt, base64_image) }), 200
    elif llm_model == "OpenAI":
        return jsonify({ "response": request_openai_model(prompt, base64_image) }), 200
    elif llm_model == "Gemini":
        return jsonify({ "response": request_gemini_model(prompt, base64_image) }), 200

@flask_app.route('/generate-chat-response', methods=['POST'])
def generate_chat_response():

    base_path = os.path.dirname(os.path.abspath(__file__))

    data = request.json

    nn_model = data["nn_model"]
    llm_model = data["llm_model"]
    prediction = data["prediction"]
    confidence = data["confidence"]
    report = data["report"]
    question = data["question"]
    history = data["history"]
    
    image_path = os.path.join(base_path, "images", nn_model, data["file_name"])
    base64_image = encode_image(image_path)

    conversation_history = "".join([f"{message['role']}: {message['content']}\n" for message in history])

    prompt = f"""
    You are an experienced neurologist specializing in oncology with expertise in analyzing MRI scans to diagnose brain tumors.
    A patient's MRI scan has been uploaded for your assessment.

    The analysis is assisted by an advanced neural network model that has made a preliminary prediction. Here are the details:

    - **Model Prediction:** The model identified the tumor as '{prediction}'.
    - **Confidence Level:** The model is {confidence}% confident in this prediction.
    - **Doctor Report:** {report}

    **Conversation History (for context):**
    The following messages are part of an ongoing conversation with the patient. Use this history to interpret the patient's current question in context, 
    determining whether it relates to the MRI scan or medical case at hand. Ignore any unrelated or general questions (e.g., "what is a Turing machine") 
    and answer only if the question is directly relevant to the medical case.

    {conversation_history}

    **Patient's Question:** {question}

    Please provide a thoughtful and empathetic response based on your medical knowledge. Your tone should reflect the care and understanding of a compassionate doctor addressing a patient, avoiding any robotic or overly technical phrasing.

    Take into account the model's prediction, the doctor's explanation, and the conversation history, but do not reference any of this information explicitly in your response. Ensure your words are supportive and patient-friendly.

    **Your response should be limited to 2 sentences.**

    **If using $, make sure to escape it by writing \\$.**
    """


    if llm_model == "Groq":
        return jsonify({ "response": request_groq_model(prompt, base64_image) }), 200
    elif llm_model == "OpenAI":
        return jsonify({ "response": request_openai_model(prompt, base64_image) }), 200
    elif llm_model == "Gemini":
        return jsonify({ "response": request_gemini_model(prompt, base64_image) }), 200

if __name__ == '__main__':

    flask_app.run(host='0.0.0.0', port=8000)