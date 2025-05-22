import streamlit as st
from pages.welcome import welcome
from pages.homePage import homePage

# Correct CSS for full-page white background in Streamlit
page_bg_css = """
<style>
    .stApp {
        background-color: white !important;
    }
</style>
"""
st.markdown(page_bg_css, unsafe_allow_html=True)

# Initiate session state for empty page
if "page" not in st.session_state:
    st.session_state.page = "Welcome"

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Welcome", "Detect Image"], index=["Welcome", "Detect Image"].index(st.session_state.page))

# Update session state page
st.session_state.page = page

# Render page
if st.session_state.page == "Welcome":
    welcome()
elif st.session_state.page == "Detect Image":
    homePage()

