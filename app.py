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

    url = 'https://drive.google.com/file/d/1VHnVoB7DFqdNHrsSN0qj99usCKUx38P9/view?usp=drive_link'
    output = 'modelo_panoramica_v1.tflite'
    gdown.download(url, output)

    interpreter = tf.lite.Interpreter(model_path='modelo_panoramica_v1.tflite')
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

    interpreter.set_tensor(input_details[0]['index'],image)

    interpreter.invoke()

    outpu_data = interpreter.get_tensor(output_details[0]['index'])
    classes = ['Panoramicas','Periapical_radiologia']

    df = pd.DataFrame()
    df['classes'] = classes
    df['probabilidades (%)'] = 100*outpu_data[0]

    fig = px.bar(df, y='classes', x='probalidades (%)', orientation='h', text='probalidades (%)',
                 title='Probalidade de classes de radiologia')
    
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
    #if image is not None:

        #previsao(interpreter,image)



if __name__ == "__main__":
    main()
