import streamlit as st
import gdown
import tensorflow as tf
import io
import pandas as pd
import numpy as np
import plotly.express as px
#import cv2
from PIL import Image


@st.cache_resource

def carrega_modelo():

    url = 'https://drive.google.com/uc?id=1VHnVoB7DFqdNHrsSN0qj99usCKUx38P9'
    output = 'modelo_panoramica_v1.tflite'
    gdown.download(url, output)

    interpreter = tf.lite.Interpreter(model_path=output)
    interpreter.allocate_tensors()

    return interpreter


def carrega_imagem():
    uploaded_file = st.file_uploader('Escolha uma imagem', type=['jpg','jpeg','png']) 

    if uploaded_file is not None:
        image_data = uploaded_file.read()
        image = Image.open(io.BytesIO(image_data))   

        st.image(image)
        st.success('Imagem foi carregada com sucesso')

        image = np.array(image,dtype=np.float32)
        image = image / 255.0
        image = np.expand_dims(image, axis=0)

        return image

def previsao(interpreter, image):

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    st.write("Detalhes de Entrada (Input Details):")
    st.json(input_details)

    st.write("Detalhes de Saída (Output Details):")
    st.json(output_details)
    
    interpreter.set_tensor(input_details[0]['index'],image) 
    
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])

    classes = ['Panoramicas','Periapical_radiologia']

    df = pd.DataFrame()
    df['classes'] = classes
    df['probabilidades (%)'] = 100*output_data[0]
    
    fig = px.bar(df,y='classes',x='probabilidades (%)',  orientation='h', text='probabilidades (%)', title='Probabilidade de Classes de Radiografia')
    
    st.plotly_chart(fig)

'''def preprocess_image(image):
    # Redimensiona a imagem para 200x220 pixels
    image = cv2.resize(image, (220, 200))
    
    # Se a imagem for RGB, converta para o formato esperado (por exemplo, BGRA ou RGBA)
    # Certifique-se de que tem 4 canais de cor, se necessário.
    if image.shape[-1] != 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGRA)  # Alterar conforme necessário
    
    # Normaliza a imagem, caso necessário (por exemplo, de 0-255 para 0-1)
    image = image.astype(np.float32)  # Converte para float32
    
    # Expande a dimensão para (1, 200, 220, 4)
    image = np.expand_dims(image, axis=0)
    
    return image'''

def main():
    
    st.set_page_config(
    page_title="Classifica radiologia",
    )

    st.write("# Classifica radiologia!")

    #carrega modelo
    interpreter = carrega_modelo()
    #carrega imagem
    image = carrega_imagem()
    #classifica
    if image is not None:

        #converter a imagem para o formato necessário
        #imagem_formatada = preprocess_image(image)
        previsao(interpreter,image)

if __name__ == "__main__":
    main()
