import os
import numpy as np
import pandas as pd
from model_yolo import get_prediction  # Assuming this function loads the YOLO model and makes predictions

def videoToFrames(path):
    # Ensure the 'videos' directory exists
    if not os.path.exists('videos'):
        os.makedirs('videos')

    # Extract the base name of the video file (without extension) for folder naming
    namePath = os.path.splitext(os.path.basename(path))[0]
    print(namePath)

    # Create a folder with the name of the video file inside the 'videos' directory
    destination_folder = os.path.join('videos', namePath)
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Create an output directory for frames inside the destination folder
    output_directory = os.path.join(destination_folder)
    os.makedirs(output_directory, exist_ok=True)

    # Use ffmpeg to extract frames from the video at 1 frame per second
    ffmpeg_command = f"ffmpeg -i \"{path}\" -vf fps=1 \"{output_directory}/video%d.jpg\""
    os.system(ffmpeg_command)

    return output_directory

def iterateFrame(path, model_path):
    # Use YOLO model to make predictions
    predictions = get_prediction(path, model_path, 0)  # Adjust parameters as necessary

    # Process YOLO predictions (e.g., class names and bounding boxes)
    detected_objects = []
    for prediction in predictions:
        class_name = prediction['class']  # Adjust this based on YOLO output format
        detected_objects.append(class_name)

    # Combine detected objects into a string
    objectsDetected = len(detected_objects)
    Objectspredictions = " ".join(detected_objects)

    # Extract frame number from file name
    nameVideo = os.path.basename(path)
    nameVideo = nameVideo.split(".")[0]
    nameVideo = nameVideo.split("video")[1]
    
    return (nameVideo, objectsDetected, Objectspredictions)

# Function to convert seconds to hh:mm:ss format
def seconds_to_hhmmss(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f'{int(hours):02}:{int(minutes):02}:{int(secs):02}'

def generate_report(path):
    model_path = 'models/best.pt'  # YOLO model path

    # Convert the video to frames
    output_dir = videoToFrames(path)

    # Iterate over the frames and make predictions
    predictions = []

    for frame in os.listdir(output_dir):
        frame_path = os.path.join(output_dir, frame)
        prediction = iterateFrame(frame_path, model_path)
        predictions.append(prediction)

    # Save the predictions to a CSV file
    df = pd.DataFrame(predictions,
                      columns=['Segundo', 'ObjectsDetected', 'predictions'])

    # Delete the rows where 'ObjectsDetected' is 0
    df = df[df.ObjectsDetected != 0]
    # Delete the 'ObjectsDetected' column
    del df['ObjectsDetected']

    # Sort the values by 'Segundo'
    df = df.sort_values(by='Segundo')

    # Convert 'Segundo' column to numeric
    df['Segundo'] = pd.to_numeric(df['Segundo'])
    # Apply the function to the 'seconds' column
    df['time'] = df['Segundo'].apply(seconds_to_hhmmss)
    df = df.drop(columns=['Segundo'])

    # Save to CSV
    output_csv_path = 'predictions.csv'
    df.to_csv(output_csv_path, index=False)
    print(f"Predictions saved to '{output_csv_path}'")
    return output_csv_path
