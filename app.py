import streamlit as st
import pickle
import numpy as np

pipe = pickle.load(open('random_forest.pkl','rb'))
df = pickle.load(open('data.pkl','rb'))


st.title("Laptop Predictor")

Brand = st.selectbox('Brand',df['Brand'].unique())

RAM = st.selectbox('RAM',df['RAM'].unique())

Operating_system = st.selectbox('Operating_system', df['Operating_system'].unique())

Processor = st.selectbox('Type of Processor',df['Processor'].unique())

Version = st.selectbox('Version',df['Version'].unique())

if st.button('Predict Price'):
     query = np.array([RAM,Brand,Processor,Operating_system,Version])
     #query = query.reshape(1,7)
     st.title(pipe.predict(query))