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
          #https://drive.google.com/file/d/1jxwhxLYwmuSNOCLgQ8h46MHpyDpPeQ9o/view?usp=drive_link
    url = 'https://drive.google.com/uc?id=1jxwhxLYwmuSNOCLgQ8h46MHpyDpPeQ9o'
    output = 'modelo_quantizado16bits.tflite'
    
    gdown.download(url, output)

    interpreter = tf.lite.Interpreter(model_path=output)
    interpreter.allocate_tensors()

    return interpreter

def carrega_imagem():
    uploaded_file = st.file_uploader('Escolha uma Radiografia', type=['jpg','jpeg','png']) 

    if uploaded_file is not None:
        image_data = uploaded_file.read()
        image = Image.open(io.BytesIO(image_data))   

        st.image(image)
        st.success('Imagem foi carregada com sucesso')

        # Redimensiona a imagem para 220x200 (necessário para o modelo)
        image = image.resize((200, 200))

        # Converte a imagem para RGBA (adiciona o canal alfa, se necessário)
        # image = image.convert("RGBA")

        image = np.array(image,dtype=np.float32)
        image = image / 255.0
        image = np.expand_dims(image, axis=0)

        return image

def previsao(interpreter, image):

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    #st.write("Detalhes de Entrada (Input Details):")
    #st.json(input_details)

    #st.write("Detalhes de Saída (Output Details):")
    #st.json(output_details)
    
    interpreter.set_tensor(input_details[0]['index'],image) 
    
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])

    classes = ['MIE', 'MID', 'PMSE', 'MSE', 'MSD', 'PMSD', 'IS', 'PMIE', 'PMID', 'Panoramicas', 'II', 'CSD', 'CSE', 'CID', 'CIE']

    df = pd.DataFrame()
    st.write("Probabilidades por Classe:")
    st.json(df.set_index('classes')['probabilidades (%)'].round(2).to_dict())
    
    df['classes'] = classes
    df['probabilidades (%)'] = 100*output_data[0]
    
    fig = px.bar(df,y='classes',x='probabilidades (%)',  orientation='h', text='probabilidades (%)', title='Probabilidade de Classes de Radiografia')
    
    st.plotly_chart(fig)


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
        previsao(interpreter,image)

if __name__ == "__main__":
    main()
