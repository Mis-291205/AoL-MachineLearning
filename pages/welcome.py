import streamlit as st
from PIL import Image
from importlib import import_module
from pages.homePage import homePage

# Inject custom CSS for white background
page_bg_css = """
<style>
    body {
        background-color: white !important;
    }
</style>
"""
st.markdown(page_bg_css, unsafe_allow_html=True)

def welcome():
    # Load gambar
    image = Image.open("./assets/welcome.png")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, use_container_width=True)

    st.markdown("""
    <style>
    @font-face {
    font-family: 'Poppins';
    src: url('./assets/fonts/Poppins-Regular.ttf') format('truetype');
    }
    .custom-text {
    font-family: 'Poppins', sans-serif;
    font-size: 24px;
    color: #333;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<p class="Poppins" style="font-size:50px; text-align: center;">Welcome to the Waste Classification</p>', unsafe_allow_html=True)

    st.markdown("""
    <p class="Poppins" 
    style="
    font-size: 20px;
    text-align: center;
    color: black;"
    >In an age where sustainability and environmental awareness are more critical than ever, proper waste classification plays a vital role in preserving our planet. This application leverages artificial intelligence to help users easily identify whether waste is organic or recyclable, promoting better sorting habits and supporting eco-friendly practices.</p>"""
    , unsafe_allow_html=True)

    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');

    .stButton > button {
    font-family: "Poppins", sans-serif;
    font-weight: 600;
    background-color: #263A6E;
    color: white;
    font-size: 32px;
    padding: 10px 24px;
    cursor: pointer;
    border: none;
    border-radius: 8px;
    }

    .stButton > button:hover {
    background-color: #39559F 
    }
    </style>
    """, unsafe_allow_html=True)

    if st.button("Continue"):
        st.session_state.page = "Detect Image"
        st.rerun()
