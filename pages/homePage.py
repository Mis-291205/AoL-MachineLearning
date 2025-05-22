import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os
from huggingface_hub import hf_hub_download

# Inject custom CSS for white background
page_bg_css = """
<style>
    body {
        background-color: white !important;
    }
</style>
"""
st.markdown(page_bg_css, unsafe_allow_html=True)

def homePage():
    def load_model(model_path):
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter

    def preprocess_input(x):
        x = x[..., ::-1]
        mean = [103.939, 116.779, 123.68]
        x[..., 0] -= mean[0]
        x[..., 1] -= mean[1]
        x[..., 2] -= mean[2]
        return x
    
    def preprocess_image(image, target_size):
        img = image.resize(target_size)
        img_array = np.array(img)
        if img_array.shape[-1] == 4:
            img_array = img_array[..., :3]
        img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
        img_array = preprocess_input(img_array)
        return img_array

    def predict_image(interpreter, image):
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        input_shape = input_details[0]['shape']
        target_size = (input_shape[1], input_shape[2])

        input_data = preprocess_image(image, target_size)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])

        if output_data.shape[-1] == 1:
            prob_recyclable = output_data[0][0]
        else:
            prob_recyclable = np.max(output_data)

        threshold = 0.5
        predicted_class_index = int(prob_recyclable > threshold)
        class_names = ['Organic', 'Recyclable']
        confidence = prob_recyclable if predicted_class_index == 1 else 1 - prob_recyclable

        return class_names[predicted_class_index], confidence

    def capture_image_from_camera():
        picture = st.camera_input("Ambil gambar menggunakan kamera")
        if picture:
            image = Image.open(picture)
            return image
        return None

    # --- LOAD MODEL FROM HUGGING FACE HUB ---
    HF_REPO_ID = "migz117/Waste-VGG16"
    MODEL_FILENAME = "model.tflite"

    interpreter = None 

    try:
        with st.spinner(f"Loading model '{MODEL_FILENAME}' from Hugging Face Hub... (This may take a moment on the first run)"):
            downloaded_model_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=MODEL_FILENAME
            )
        
        interpreter = load_model(downloaded_model_path) 

    except Exception as e:
        st.error(f"An error occurred while trying to load the model from Hugging Face Hub: {e}")

    try:
        # Show a message while downloading/loading
        with st.spinner(f"Loading model '{MODEL_FILENAME}' from Hugging Face Hub... (This may take a moment on the first run)"):
            downloaded_model_path = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=MODEL_FILENAME
                # Optional: specify a local cache directory if you prefer
                # cache_dir="./my_hf_models_cache/"
            )
        
        # Use your existing load_model function with the downloaded path
        interpreter = load_model(downloaded_model_path) 

        if interpreter:
            st.success(f"Model '{MODEL_FILENAME}' loaded successfully from Hugging Face Hub.")
        else:
            # load_model would have printed an error if tf.lite.Interpreter failed.
            # This is an additional check.
            st.error("Failed to initialize TFLite interpreter after downloading model.")

    except Exception as e:
        st.error(f"An error occurred while trying to load the model from Hugging Face Hub: {e}")
        st.error(f"Details: Repo ID='{HF_REPO_ID}', Filename='{MODEL_FILENAME}'.")
        st.error("Please check internet connection, model name, and that 'huggingface-hub' is installed.")
    # --- END OF MODEL LOADING ---

    st.title("Waste Type Detector")

    # Initiate session state
    if 'use_camera' not in st.session_state:
        st.session_state.use_camera = False
    if 'uploaded' not in st.session_state:
        st.session_state.uploaded = None

    # Reset button
    if st.button("Reset", key="reset"):
        st.session_state.use_camera = False
        st.session_state.uploaded = None
        st.rerun()

    # Activate camera state
    if st.button("Aktifkan Kamera", key="activate_camera"):
        st.session_state.use_camera = True
        st.session_state.uploaded = None 

    # Image upload state
    if not st.session_state.use_camera:
        uploaded_image = st.file_uploader("Unggah gambar", type=["png", "jpg", "jpeg"])
        if uploaded_image is not None:
            st.session_state.uploaded = uploaded_image

    image = None

    # Take photo
    if st.session_state.use_camera:
        camera_photo = st.camera_input("Ambil gambar dari kamera")
        if camera_photo is not None:
            image = Image.open(camera_photo)
            st.image(image, caption="Gambar dari Kamera", use_container_width=True)

    # Get uploaded image
    elif st.session_state.uploaded is not None:
        image = Image.open(st.session_state.uploaded)
        st.image(image, caption="Gambar yang diunggah", use_container_width=True)

    # Predict image
    if image is not None:
        predicted_label, prediction = predict_image(interpreter, image)
        if predicted_label == "Organic":
            st.markdown(f"""
                <h2 style='color: green;
                font-size: 42px;
                text-align: center;'
                >Result: {predicted_label}</h2>""", unsafe_allow_html=True)
            st.markdown("""
                <h2 style='color: green;
                font-size: 28px;
                text-align: center;'
                >Your image is organic waste</h2>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <h2 style='color: red;
                font-size: 42px;
                text-align: center;'
                >Result: {predicted_label}</h2>""", unsafe_allow_html=True)
            st.markdown("""
                <h2 style='color: red;
                font-size: 28px;
                text-align: center;'
                >Your image is recyclable waste</h2>""", unsafe_allow_html=True)

        st.markdown("""
            <p style='text-align: center; font-style: italic; color: gray;'>
            *Waste Type Detector is not 100% accurate, please recheck and validate the result.
            </p>""", unsafe_allow_html=True)

