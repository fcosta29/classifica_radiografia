import streamlit as st
import os
import boto3
import gdown
import hashlib
import tensorflow as tf
import io
import pandas as pd
import numpy as np
import plotly.express as px
from PIL import Image
import imagehash
import cv2


@st.cache_resource

def calcular_similaridade_hash_arquivo_local(caminho):
    imagem = Image.open(caminho)
    return imagehash.phash(imagem)  # ou dhash, ahash

def calcular_hash_arquivo_local(caminho):
    """Calcula o hash MD5 de um arquivo local."""
    hash_md5 = hashlib.md5()
    with open(caminho, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def calcular_hash_bytes(dados_bytes):
    """Calcula o hash MD5 de dados binários."""
    return hashlib.md5(dados_bytes).hexdigest()

def calcular_similaridade_hash_bytes(dados_bytes):
    imagem = Image.open(io.BytesIO(dados_bytes))
    return imagehash.phash(imagem)

def comparar_imagem_caminho_com_bytes(img_path, img_bytes, tamanho=(200, 200), limite_iguais=0.95, limite_semelhantes=0.90):
    # Lê a imagem do caminho
    img1 = cv2.imread(img_path)
    if img1 is None:
        raise ValueError("Imagem do caminho não pôde ser carregada.")

    # Lê a imagem a partir dos bytes
    nparr = np.frombuffer(img_bytes, np.uint8)
    img2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img2 is None:
        raise ValueError("Imagem dos bytes não pôde ser decodificada.")

    # Redimensiona ambas para o mesmo tamanho
    img1 = cv2.resize(img1, tamanho)
    img2 = cv2.resize(img2, tamanho)

    # Calcula a diferença absoluta e converte para escala de cinza
    diff = cv2.absdiff(img1, img2)
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Soma os valores das diferenças
    score = np.sum(diff_gray)
    max_diff = tamanho[0] * tamanho[1] * 255
    similaridade = 1 - (score / max_diff)

    # Classificação com base nos limites
    if similaridade >= limite_iguais:
        resultado = "iguais"
    elif similaridade >= limite_semelhantes:
        resultado = "semelhantes"
    else:
        resultado = "diferentes"

    return resultado, round(similaridade, 2)

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
    df['classes'] = classes
    df['probabilidades (%)'] = 100*output_data[0]

    #st.write("Probabilidades por Classe:")
    #st.json(df.set_index('classes')['probabilidades (%)'].round(2).to_dict())
    
    
    fig = px.bar(df,y='classes',x='probabilidades (%)',  orientation='h', text='probabilidades (%)', title='Probabilidade de Classes de Radiografia')
    
    st.plotly_chart(fig)

def valida_imagem_duplicada(image_upload):

    aws_key = st.secrets["AWS_KEY"]
    aws_secret = st.secrets["AWS_SECRET"]
    #key = os.getenv('AWS_KEY')
    #secret = os.getenv('AWS_SECRET')

    st.write("key de acesso ao AWS")
    st.write(aws_key)

    bucket_name = "brzd-dev-images"
    s3 = boto3.client(
        's3',
        aws_access_key_id=aws_key,
        aws_secret_access_key=aws_secret
    )

    try:
        response = s3.list_objects_v2(Bucket=bucket_name)
        if 'Contents' in response:
            # Calcula hash da imagem local
            local_hash = calcular_hash_arquivo_local(image_upload)
            local_similaridade = calcular_similaridade_hash_arquivo_local(image_upload)

            st.write("Listando arquivos no AWS:")

            for obj in response['Contents']:
                if obj['Key'].endswith('.jpg'):

                    st.write(f"Nome do arquivo: {obj['Key']}")    
                    
                    # Baixa imagem do S3 para memória
                    #obj_data = s3.get_object(Bucket=bucket_name, Key=obj['Key'])
                    #s3_image_bytes = obj_data['Body'].read()

                    # Calcula hash da imagem do S3
                    #s3_hash = calcular_hash_bytes(s3_image_bytes)
                    # Calcula similaridade da imagem do S3
                    #s3_similaridade = calcular_similaridade_hash_bytes(s3_image_bytes)
     
                    # Compara
                    #if s3_hash == local_hash:
                        #print(f"IMAGEM IGUAL: {obj['Key']}")
                        #break
                    #else:                     
                        #if local_similaridade == s3_similaridade:
                            #print(f"IMAGEM IGUAL: {obj['Key']}")
                        #else:

                            #resultado, porcentagem = comparar_imagem_caminho_com_bytes(image_upload, s3_image_bytes) 

                            #if resultado == "iguais":
                                #print(f"IMAGEM IGUAL: {obj['Key']}, similaridade: {porcentagem}")
                                #break
                            #else:
                                #print(f"IMAGEM DIFERENTE: {obj['Key']}, similaridade: {porcentagem}")

        else:
            print("Nenhuma imagem encontrada no bucket.")
    except Exception as e:
        print(f"Erro: {e}")

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
        valida_imagem_duplicada(image)

if __name__ == "__main__":
    main()
