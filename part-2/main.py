from tensorflow.keras.preprocessing import image
import streamlit as st
import numpy as np
import requests

# Saliency Maps 

def draw_saliency_maps(nn_models, uploaded_file):

    img = image.load_img(uploaded_file, target_size=(299, 299))

    cols = st.columns(len(nn_models) + 1)

    with cols[0]:
        st.image(img, caption="Original Image", use_container_width=True)

    for idx, nn_model in enumerate(nn_models):

        with cols[idx + 1]:

            st.image(np.array(nn_model["saliency_map"]), caption=f"{nn_model['name']}", use_container_width=True)

# Classification Results

def get_prediction_color(prediction):

    color_palette = {
        "No tumor": "#2E3D7C",
        "Meningioma": "#BA292E",
        "Glioma": "#E15D3A", 
        "Pituitary": "#FFA73C",
    }

    return color_palette.get(prediction, "#282528")

def draw_classification_results(nn_models):

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

    cols = st.columns(len(nn_models))

    # Display each model's information in a card layout
    for idx, nn_model in enumerate(nn_models):
        
        with cols[idx]:

            sorted_indices = np.argsort(nn_model['probability_distribution'])[::-1]

            label = labels[sorted_indices[0]]
            probability = nn_model['probability_distribution'][sorted_indices[0]]

            probability_color = get_prediction_color(label)

            st.markdown(
                f"""
                <div class="model-card">
                    <div class="model-name">{nn_model['name']}</div>
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
                    probability = nn_model['probability_distribution'][index]

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

    nn_models = list()

    if "Xception" in selected_models and st.session_state.get("Xception") is None:
        
        response = requests.post("https://selected-gently-swift.ngrok-free.app/run_xception", files={
            "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
        }).json()

        st.session_state["Xception"] = response
        nn_models.append(response)

    elif "Xception" in selected_models and st.session_state.get("Xception") is not None:
        
        nn_models.append(st.session_state.get("Xception"))
    
    if "ResNet50" in selected_models and st.session_state.get("ResNet50") is None:
       
        response = requests.post("https://selected-gently-swift.ngrok-free.app/run_resnet", files={
            "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
        }).json()

        st.session_state["ResNet50"] = response
        nn_models.append(response)

    elif "ResNet50" in selected_models and st.session_state.get("ResNet50") is not None:
        
        nn_models.append(st.session_state.get("ResNet50"))
    
    if "Custom CNN" in selected_models and st.session_state.get("Custom CNN") is None:
        
        response = requests.post("https://selected-gently-swift.ngrok-free.app/run_cnn", files={
            "file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)
        }).json()

        st.session_state["Custom CNN"] = response
        nn_models.append(response)

    elif "Custom CNN" in selected_models and st.session_state.get("Custom CNN") is not None:
        
        nn_models.append(st.session_state.get("Custom CNN"))

    if selected_models:
        
            st.write("## Saliency Maps")

            draw_saliency_maps(nn_models, uploaded_file)

            st.write("## Classification Results")

            draw_classification_results(nn_models)

            llm_model = st.selectbox(
                "LLM Model",
                options=["Groq", "OpenAI", "Gemini"]
            )

            if llm_model is not None:
                
                st.write("## Report")

                for nn_model in nn_models:

                    if st.session_state.get(f"{nn_model['name']}_{llm_model}_report") is None:
                        
                        report = requests.post("https://selected-gently-swift.ngrok-free.app/generate-explanation", json={
                            "nn_model": nn_model["name"],
                            "llm_model": llm_model,
                            "file_name": uploaded_file.name,
                            "prediction": nn_model["prediction"],
                            "confidence": nn_model["confidence"]
                        }).json()

                        st.session_state[f"{nn_model['name']}_{llm_model}_report"] = report["response"]
                    
                    st.write(f"### {nn_model['name']}")
                    st.write(st.session_state.get(f"{nn_model['name']}_{llm_model}_report"))

                    with st.expander("Chat with LLM"):
                        
                        chat_container = st.container(height=300)

                        if st.session_state.get(f"{nn_model['name']}_{llm_model}_history") is None:
                            st.session_state[f"{nn_model['name']}_{llm_model}_history"] = list()

                        question = st.chat_input("Say something", key=f"{nn_model['name']}_{llm_model}_question") 

                        if st.button("Clear Chat History", key=f"{nn_model['name']}_{llm_model}_clear_history"):
                            st.session_state[f"{nn_model['name']}_{llm_model}_history"] = []

                        with chat_container:
                            
                            if question:
                                
                                st.session_state[f"{nn_model['name']}_{llm_model}_history"].append({"role": "user", "content": question})
                                
                                with st.spinner("Waiting for response..."):
                                    response = requests.post("https://selected-gently-swift.ngrok-free.app/generate-chat-response", json={
                                        "llm_model": llm_model,
                                        "nn_model": nn_model["name"],
                                        "file_name": uploaded_file.name,
                                        "prediction": nn_model["prediction"],
                                        "confidence": nn_model["confidence"],
                                        "report": st.session_state.get(f"{nn_model['name']}_{llm_model}_report"),
                                        "question": question,
                                        "history": st.session_state[f"{nn_model['name']}_{llm_model}_history"]
                                    }).json()
                                
                                st.session_state[f"{nn_model['name']}_{llm_model}_history"].append({"role": "assistant", "content": response["response"]})

                            for message in st.session_state[f"{nn_model['name']}_{llm_model}_history"]:
                                st.chat_message(message["role"]).write(message["content"]) 
