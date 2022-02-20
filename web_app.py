import requests
import streamlit as st
from PIL import Image

st.set_page_config(layout="wide")

st.title("Applied AI Project - Demo")



logo_applied_roots = Image.open("./data/webapp_logos/Applied_Roots_logo.png")
logo_uoh = Image.open("./data/webapp_logos/University_of_Hyderabad_Logo.png")

st.sidebar.title("About")
st.sidebar.info(
    " This project is done as a part of the Post Graduate Diploma \
                thesis or end semester project. \
                This short demo helps us to pick a image and get the prediction of the developed Multi class classifer "
)
st.sidebar.info("Used Dataset- CDiscount Classification Challenge.")
st.sidebar.image(logo_applied_roots, width=300)  # , use_column_width = 'auto')
st.sidebar.image(logo_uoh, width=300, use_column_width="auto")

st.write("Please select any  one of the methods below - Upload Image or Take a Picture")
image = st.file_uploader("Choose an Image")

if st.button("Open Camera"):
    image = st.camera_input("Take a picture")
if st.button("Predict Class"):
    if image is not None:
        files = {"file": image.getvalue()}
        with st.spinner("API Request Initiated.Waiting for Response"):
            res = requests.post(("http://localhost:5001/predict/"), files=files)
        st.success("Request completed Successfully")
        text = res.json()
        st.session_state.key = text
        st.image(image, caption="Uploaded Image")

    print(text)
    # st.write(text)

    predictions = text["model_response"]

    col1, col2, col3 = st.columns(3)
    col1.metric(label="Category_Index", value=predictions["category_index"])
    col2.metric(label="Confidence", value=predictions["confidence"])
    col3.metric(label="Category_id", value=predictions["category_id"])

    # col4, col5, col6 = st.columns(3)
    st.metric(label="Category_Level1", value=predictions["category_l1"])
    st.metric(label="Category_Level2", value=predictions["category_l2"])
    st.metric(label="Category_Level3", value=predictions["category_l3"])

    col4, col5 = st.columns(2)
    col4.metric(label="Response Time", value=text["response_time"])
    col5.metric(label="Model Inference time", value=text["model_inference_time"])
