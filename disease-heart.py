# Prepare the library
from typing import Any 
import streamlit as st
import pandas as pd 

import pickle
import time
from PIL import Image

#set judul halaman web
st.set_page_config(page_title="Final Project", layout="wide")

# Center-aligned text using CSS
st.markdown("""
    <h1 style='text-align: center;'>Welcome to Heart Disease Machine Learning Dashboard</h1>
    """, unsafe_allow_html=True)

add_selectitem = st.sidebar.selectbox("Want do you want?", ("Checking your heart!",))

#buat logo heart disease ke tengah
col1, col2, col3 = st.columns(3)

with col1:
    st.write(' ')

with col2:
    st.image('heartdisease.jpg', width=350)

with col3:
    st.write(' ')

# Center-aligned text using Streamlit
st.write("""
    Heart disease refers to a group of conditions that can affect the heart's functioning. This includes coronary artery disease, heart failure, arrhythmias, and many other conditions. Heart disease can be life-threatening and is often associated with risk factors such as high blood pressure, high cholesterol levels, poor dietary habits, smoking, lack of physical activity, and genetic factors. 
    """)

st.write("""
- Age (Patient's Age): This variable records the age of the patient in years. Age can be a significant risk factor in the development of heart disease, as the risk of heart disease tends to increase with advancing age.

- Sex (Gender): This variable indicates the gender of the patient, with a value of 1 for male and 0 for female. Some studies suggest that the risk of heart disease may differ between males and females, with males often having a higher risk.

- CP (Chest Pain Type): This variable describes the type of chest pain experienced by the patient. It has four category values:
Value 1: Typical Angina (Stable Angina)
Value 2: Atypical Angina (Unstable Angina)
Value 3: Non-Anginal Pain
Value 4: Asymptomatic (Without Symptoms)
Chest pain is a common symptom that can be an indicator of heart problems.

- Thalach (Maximum Heart Rate): This is the maximum heart rate of the patient. A high maximum heart rate can indicate good heart performance.

- Exang (Exercise-Induced Angina): This is a binary variable indicating whether the patient experiences angina during exercise. A value of 1 indicates the presence of angina, while 0 indicates the absence of angina.

- Oldpeak (ST Depression): ST depression is a change on an electrocardiogram (EKG) that occurs during a stress test relative to rest. This variable measures the extent of ST depression.

- Slope: This is a variable that describes the slope of the EKG segment during a stress test. It can have three values: 0 (downsloping), 1 (flat), and 2 (upsloping).

- CA (Number of Major Vessels): This is the number of major blood vessels with significant narrowing (0-3). The number of narrowed blood vessels can be an indicator of coronary heart disease.

- Thal (Thallium Scan Result): This variable represents the result of a thallium scan test with three categories: 1 (normal), 2 (fixed defect), and 3 (reversible defect).
""")

# Collects user input features into dataframe
st.sidebar.header('User Input Features:')
uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
        def user_input_features():
            st.sidebar.header('Manual Input')
            cp = st.sidebar.slider('Chest pain type', 1,4,2)
            if cp == 1.0:
                wcp = "Typical angina"
            elif cp == 2.0:
                wcp = "Atypical angina"
            elif cp == 3.0:
                wcp = "Non angina"
            else:
                wcp = "Asymptomatic"
            st.sidebar.write("Type of Chest pain : ", wcp)
            thalach = st.sidebar.slider("Maximum heart rate achieved", 71, 202, 80)
            slope = st.sidebar.slider("Slope of the peak exercise ST segment", 0, 2, 1)
            oldpeak = st.sidebar.slider("ST depression induced", 0.0, 6.2, 1.0)
            exang = st.sidebar.slider("Exercise induced angina", 0, 1, 1)
            ca = st.sidebar.slider("Number of major vessels", 0, 3, 1)
            thal = st.sidebar.slider("Result of thallium test", 1, 3, 1)
            sex = st.sidebar.selectbox("Sex", ('Female', 'Male'))
            if sex == "Female":
                sex = 0
            else:
                sex = 1 
            age = st.sidebar.slider("Age", 29, 77, 30)
            data = {'cp': cp,
                    'thalach': thalach,
                    'slope': slope,
                    'oldpeak': oldpeak,
                    'exang': exang,
                    'ca':ca,
                    'thal':thal,
                    'sex': sex,
                    'age':age}
            features = pd.DataFrame(data, index=[0])
            return features
        input_df = user_input_features()


# Loading images
diseaseheart = Image.open('heartdisease.jpeg')
healthheart = Image.open('niceheart.jpg')

loaded_model = None
output = " "

if st.sidebar.button('Click Here To Predict'):
    df = input_df
    st.write(df)
    with open("best_model_lr.pkl", 'rb') as file:  
        loaded_model = pickle.load(file)

if loaded_model is not None:
    prediction = loaded_model.predict(df)
    if (prediction == 0).any():
        result = 'No Heart Disease'
    else:
        result = 'Yes Heart Disease'

    output = str(result)

    st.subheader('Prediction Result: ')

    with st.spinner('Wait for it...'):
        time.sleep(4)

    st.success(f"Result of this Prediction is {output}")

    if (prediction == 0).any():
        st.image(healthheart, width=400)
    else:
        st.image(diseaseheart, width=400)
