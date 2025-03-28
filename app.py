import streamlit as st
import gdown
import tensorflow as tf
import io
import pandas as pd
import numpy as np
import plotly.express as px
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

def preprocess_image(image):
    # Abre a imagem com PIL
    pil_image = Image.open(io.BytesIO(image.read()))
    
    # Redimensiona a imagem para 200x220
    pil_image = pil_image.resize((220, 200))
    
    # Converte a imagem para RGBA (se for RGB, adiciona um canal alfa)
    pil_image = pil_image.convert("RGBA")
    
    # Converte a imagem para um numpy array e normaliza os valores
    image_array = np.array(pil_image, dtype=np.float32)
    image_array /= 255.0  # Normaliza os valores para o intervalo [0, 1]
    
    # Adiciona a dimensão do batch (1, 200, 220, 4)
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array    

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
        image_reprocessada = preprocess_image(image)
        previsao(interpreter,image_reprocessada)

if __name__ == "__main__":
    main()
