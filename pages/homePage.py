import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os  # <--- IMPORT THE OS MODULE HERE

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
    def load_model(model_path_to_load): # Renamed parameter for clarity
        # Debug: Print the path that is being used
        st.write(f"Attempting to load model from: {model_path_to_load}")
        if not os.path.exists(model_path_to_load):
            st.error(f"MODEL FILE NOT FOUND at: {model_path_to_load}")
            # More debug info if file not found
            script_dir_debug = os.path.dirname(os.path.abspath(__file__))
            project_root_debug = os.path.abspath(os.path.join(script_dir_debug, ".."))
            st.info(f"Script directory: {script_dir_debug}")
            st.info(f"Calculated project root: {project_root_debug}")
            if os.path.exists(project_root_debug):
                st.info(f"Contents of project root: {os.listdir(project_root_debug)}")
            assets_dir_debug = os.path.join(project_root_debug, "assets")
            if os.path.exists(assets_dir_debug):
                st.info(f"Contents of 'assets' directory: {os.listdir(assets_dir_debug)}")
            else:
                st.warning("'assets' directory does not exist at the calculated project root.")
            return None # Stop if file not found

        try:
            interpreter = tf.lite.Interpreter(model_path=model_path_to_load)
            interpreter.allocate_tensors()
            return interpreter
        except Exception as e:
            st.error(f"Error initializing TFLite interpreter: {e}")
            return None


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
            prob_recyclable = np.max(output_data) # Or consider np.argmax if it's multi-class probabilities

        threshold = 0.5
        predicted_class_index = int(prob_recyclable > threshold)
        class_names = ['Organic', 'Recyclable'] # Ensure these match your model's output logic
        # Corrected confidence calculation
        if predicted_class_index == 1: # Recyclable
            confidence = prob_recyclable
        else: # Organic
            confidence = 1 - prob_recyclable
        
        return class_names[predicted_class_index], confidence

    # --- CORRECTED MODEL PATH LOADING ---
    # Get the directory of the current script (homePage.py)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up one level to the project root (e.g., /mount/src/aol-machinelearning/)
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    # Construct the full path to the model in the 'assets' directory
    model_path = os.path.join(project_root, "assets", "model.tflite")
    # --- END OF CORRECTION ---

    interpreter = load_model(model_path)

    st.title("Waste Type Detector")

    # Initiate session state
    if 'use_camera' not in st.session_state:
        st.session_state.use_camera = False
    if 'uploaded_image_data' not in st.session_state: # Renamed to avoid confusion with widget
        st.session_state.uploaded_image_data = None
    if 'show_results' not in st.session_state:
        st.session_state.show_results = False


    col1, col2 = st.columns(2)

    with col1:
        if st.button("Unggah Gambar", key="upload_button", use_container_width=True):
            st.session_state.use_camera = False
            st.session_state.show_results = False # Reset results display
            # Trigger rerun to show file_uploader if it was hidden
            st.rerun()

    with col2:
        if st.button("Aktifkan Kamera", key="activate_camera", use_container_width=True):
            st.session_state.use_camera = True
            st.session_state.uploaded_image_data = None # Clear any uploaded image
            st.session_state.show_results = False # Reset results display
            st.rerun()
            
    if st.button("Reset", key="reset_all", type="primary", use_container_width=True):
        st.session_state.use_camera = False
        st.session_state.uploaded_image_data = None
        st.session_state.show_results = False
        # Clear any cached camera or uploader widgets by rerunning
        st.rerun()


    image_to_process = None
    image_caption = ""

    if st.session_state.use_camera:
        camera_photo = st.camera_input("Ambil gambar dari kamera", key="camera_widget")
        if camera_photo is not None:
            image_to_process = Image.open(camera_photo)
            image_caption = "Gambar dari Kamera"
            st.session_state.show_results = True # Show results when new image is captured
    else:
        uploaded_file_widget = st.file_uploader("Unggah gambar", type=["png", "jpg", "jpeg"], key="uploader_widget")
        if uploaded_file_widget is not None:
            st.session_state.uploaded_image_data = uploaded_file_widget # Store the file uploader object
            image_to_process = Image.open(st.session_state.uploaded_image_data)
            image_caption = "Gambar yang diunggah"
            st.session_state.show_results = True # Show results when new image is uploaded
        elif st.session_state.uploaded_image_data is not None and not uploaded_file_widget:
            pass


    if image_to_process is not None:
        st.image(image_to_process, caption=image_caption, use_container_width=True)
        
        if st.session_state.show_results and interpreter is not None:
            predicted_label, confidence = predict_image(interpreter, image_to_process) # Changed prediction to confidence
            result_color = "green" if predicted_label == "Organic" else "red"
            waste_type_text = "organic waste" if predicted_label == "Organic" else "recyclable waste"

            st.markdown(f"""
                <h2 style='color: {result_color};
                font-size: 36px; 
                text-align: center;'
                >Result: {predicted_label}</h2>""", unsafe_allow_html=True)
            st.markdown(f"""
                <h3 style='color: {result_color};
                font-size: 24px;
                text-align: center;'
                >Your image is {waste_type_text} (Confidence: {confidence:.2f})</h3>""", unsafe_allow_html=True)
        elif interpreter is None and st.session_state.show_results:
            st.error("Model is not loaded, cannot predict.")
