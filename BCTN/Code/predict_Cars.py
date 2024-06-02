import streamlit as st
import numpy as np
import pandas as pd
import base64
import requests
import joblib

def app():
    # st.header('Dự đoán')
    def sidebar_bg(img_url):
        side_bg_ext = 'jpg'  # Assuming the image format is PNG (can be adjusted if needed)

        # Retrieve image data from URL
        response = st.cache_resource(requests.get)(img_url, stream=True)
        img_data = response.content

        # Encode image data as base64
        encoded_data = base64.b64encode(img_data).decode()

        # Apply background image style to sidebar
        st.markdown(
            f"""
            <style>
            [data-testid="stSidebar"] > div:first-child {{
                background: url(data:image/{side_bg_ext};base64,{encoded_data});
                background-size: cover;  /* Adjust background sizing as needed */
            }}
            </style>
            """,
            unsafe_allow_html=True,
        )

    # Example usage with a valid image URL
    # img_url = "https://images.rawpixel.com/image_800/cHJpdmF0ZS9sci9pbWFnZXMvd2Vic2l0ZS8yMDIzLTEwL3Jhd3BpeGVsb2ZmaWNlM19taW5pbWFsX2ZsYXRfdmVjdG9yX2Flc3RoZXRpY19pbGx1c3RyYXRpb25fb2ZfYV9hYWMyODk1Ny02ODI3LTQ3OGUtOTQ2Ni0wNWI0MzVhYjk2MmQtYi5qcGc.jpg"  # Replace with your desired image URL
    img_url = "https://i.pinimg.com/originals/28/c1/d7/28c1d768683e896a84a4ec5f37f02463.jpg"  # Replace with your desired image URL
    sidebar_bg(img_url)
    
    with st.sidebar:
        message_html = """
        <style>
            .recommend-message {
                color: white; /* Change this to your desired color */
                font-weight: bold;
                font-size: 45px;
            }
        </style>

        <div class="recommend-message">PREDICT</div>
        """
        st.sidebar.write(message_html, unsafe_allow_html=True)

    # Load models and scalers
    model = joblib.load('C:\Python\Capstone\Recommend_cars\Model\knn_model.joblib')
    encoder = joblib.load('C:\Python\Capstone\Recommend_cars\Model\encoder.joblib')
    scaler = joblib.load('C:\Python\Capstone\Recommend_cars\Model\scaler.joblib')

    # Creating two columns
    col1, col2 = st.columns(2)

    # Column for technical specifications
    with col1:
        year = st.slider('Năm sản xuất', min_value=2000, max_value=2024, value=2023)
        seats = st.slider('Số chỗ ngồi', min_value=2, max_value=8, value=5)
        km = st.slider('Km đã đi', min_value=0, max_value=500000, value=50000, step=1000)
        engine = st.slider('Dung tích động cơ', min_value=1.0, max_value=5.0, value=2.0, step=0.1)

    # Column for design and style
    with col2:
        # Danh sách các hãng xe
        brands = ['KIA', 'TOYOTA', 'FORD', 'MAZDA', 'BMW', 'HONDA', 'NISSAN',
                'HYUNDAI', 'MITSUBISHI', 'VOLKSWAGEN', 'MG', 'LEXUS', 'CHEVROLET',
                'LANDROVER', 'VINFAST', 'MERCEDES-BENZ', 'SUZUKI', 'PORSCHE',
                'THACO', 'ISUZU', 'DAEWOO', 'SUBARU', 'AUDI', 'PEUGEOT', 'VOLVO',
                'BENTLEY', 'HAVAL', 'JEEP', 'ROVER', 'MINI', 'ROLLS ROYCE']
        # Sắp xếp danh sách theo thứ tự bảng chữ cái
        sorted_brands = sorted(brands)
        # Sử dụng danh sách đã sắp xếp trong selectbox
        brand = st.selectbox('Hãng xe', options=sorted_brands)
        body_style = st.selectbox('Kiểu dáng', options=['Sedan', 'SUV', 'Hatchback', 'Wagon', 'Convertible/Cabriolet', 'Coupe'])
        color = st.selectbox('Màu sắc', options=['Trắng', 'Đen', 'Bạc', 'Xanh', 'Đỏ', 'Vàng'])
        transmission = st.selectbox('Hộp số', options=['Hộp số tự động', 'Hộp số sàn'])
        fuel = st.selectbox('Nhiên liệu', options=['Xăng', 'Dầu', 'Hybrid', 'Điện'])
    # Predict button in the center or under columns
    if st.button('Dự đoán giá'):
        # Encode and predict
        features = np.array([[body_style, color, transmission, fuel,brand]])
        encoded_features = encoder.transform(features)
        features = np.array([[year, seats, km, engine]])
        all_features = np.hstack((features, encoded_features))
        all_features = scaler.transform(all_features)
        prediction = model.predict(all_features)
        st.write(f'Dự đoán giá xe: {prediction[0]:,.0f} VND')


