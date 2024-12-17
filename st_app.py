import pandas as pd
import streamlit as st
from datetime import date
import cv2
from model_yolo import get_prediction
from generate_report_yolo import generate_report
from video import generate_video
from PIL import Image
import os

image_ice = Image.open('img/logo_fuerzaaereaazulvertica.png')
image_procol = Image.open('img/aeroespacial_marca.png')

col1, col2, col3 = st.columns(3)

with col1:
    st.image(image_procol, width=150, use_column_width=True)

with col2:
    # Add content to the second column if desired
    pass

with col3:
    st.image(image_ice, width=150, use_column_width=True)

# st.header("*Kiruna*")# - Extractor de información de propiedades")
st.header("Eagle View AI")
st.write(
    "Software para reconocimienot de objetivos de interes en videos e imagenes captados por sensores aerotransportados"
)
# st.write("Bogotá y Soacha")
st.write('***')
#st.write('**Inputs**')

option = st.selectbox('Escoge un modelo:', ('modelo1', 'modelo2'))

model_path = 'models/best.pt'
otro = 0
if option == 'YOLOv1':
    model_path = 'models/best.pt'
elif option == 'YOLOv2':
    model_path = 'models/best1.pt'

st.write('You selected:', option)

# Image uploader
st.title("Sube la imagen")
uploaded_file = st.file_uploader("Escoge una imagen...",
                                 type=["jpg", "webp", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    # st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Get the directory path of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Save the uploaded image with the same name in the same folder
    uploaded_filename = os.path.join(script_dir, uploaded_file.name)
    image.save(uploaded_filename)
    print("Imagen guardada como: ", uploaded_filename)

    prediction = get_prediction(uploaded_filename, model_path, otro)
    st.write("Aquí está la predicción:")
    st.image(prediction,
             caption='Resultados de la predicción',
             use_column_width=True)

else:
    st.write(
        'No se ha subido ninguna imagen. Por favor sube una imagen para continuar.'
    )

# %% Video
st.write('***')
st.title("Generar el video de reporte")
video_file = st.file_uploader("Sube un video",
                              type=["mp4", "avi", "mov, mpeg"])

# If a video file is uploaded, display it and save it locally
if video_file is not None:
    # Display the uploaded video
    # st.video(video_file)

    # Get the file name of the uploaded video
    video_name = video_file.name

    # Define the directory where the video will be saved
    save_dir = "uploaded_videos"

    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Save the uploaded video locally
    save_path = os.path.join(save_dir, video_name)
    with open(save_path, "wb") as f:
        f.write(video_file.read())

    # Path of the video
    output_video_path = generate_video(save_path)

    # Display the video file
    st.video(output_video_path, format="video/mp4")

    # Provide a download button for the video
    with open(output_video_path, "rb") as file:
        video_bytes = file.read()
        st.download_button(
            label="Download Video",
            data=video_bytes,
            file_name=video_name,
            mime="video/mp4"
        )

# %%
# Generate report
st.write('***')
st.title("Genere el reporte")
# Allow the user to upload a video file
video_file = st.file_uploader("Sube un video", type=["mp4", "avi", "mov"])

# If a video file is uploaded, display it and save it locally
if video_file is not None:
    # Display the uploaded video
    # st.video(video_file)

    # Get the file name of the uploaded video
    video_name = video_file.name

    # Define the directory where the video will be saved
    save_dir = "uploaded_videos"

    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Save the uploaded video locally
    save_path = os.path.join(save_dir, video_name)
    with open(save_path, "wb") as f:
        f.write(video_file.read())

    # Path of the video
    # video_path = "C:/Users/juanm/Videos/VideoCodefest_007-3min.avi"
    #csv_file = generate_report(save_path)
    csv_file = generate_report(save_path)
    

    df = pd.read_csv(csv_file)

    # Display a subset of the DataFrame
    st.write("Subset of CSV File:")
    st.dataframe(df.head(20))

    # Provide a download button for the CSV file
    with open(csv_file, "rb") as file:
        csv_bytes = file.read()
        st.download_button(
            label="Download CSV",
            data=csv_bytes,
            file_name=os.path.basename(csv_file),
            mime="text/csv"
        )
