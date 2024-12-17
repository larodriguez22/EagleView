import os
import pandas as pd
from ultralytics import YOLO

def videoToFrames(path):
    if not os.path.exists('videos'):
        os.makedirs('videos')

    namePath = os.path.splitext(os.path.basename(path))[0]
    print(namePath)

    destination_folder = os.path.join('videos', namePath)
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

   
    output_directory = os.path.join(destination_folder)
    os.makedirs(output_directory, exist_ok=True)

   
    ffmpeg_command = f"ffmpeg -i \"{path}\" -vf fps=1 \"{output_directory}/frame%d.jpg\""
    os.system(ffmpeg_command)

    return output_directory

def iterateFrame(frame_path, model):
    object_detected = {}
    results = model(frame_path)
    boxes = results[0].boxes  
    names = results[0].names
    
    for box in boxes:
        coords = box.xyxy[0]  
        confidence = box.conf[0]
        class_idx = box.cls[0].item()  
        class_name = names[class_idx]
        
        if class_name in object_detected:
            object_detected[class_name] += 1
        else:
            object_detected[class_name] = 1
            
    return object_detected

# Function to convert seconds to hh:mm:ss format
def seconds_to_hhmmss(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f'{int(hours):02}:{int(minutes):02}:{int(secs):02}'

def generate_report(video_path):
    model_path = 'models/best.pt'  # YOLO model path
    
    # Convert the video to frames
    output_dir = videoToFrames(video_path)
    
    # Load YOLO model
    model = YOLO(model_path)
    
    predictions = []
    
    for frame in sorted(os.listdir(output_dir)):
        frame_path = os.path.join(output_dir, frame)
        frame_number = int(frame.split('frame')[1].split('.')[0])
        detected_objects = iterateFrame(frame_path, model)
        objects_detected = len(detected_objects)
        object_predictions = " ".join([f"{key}: {value}" for key, value in detected_objects.items()])
        predictions.append((frame_number, objects_detected, object_predictions))
    
   
    df = pd.DataFrame(predictions, columns=['Segundo', 'ObjectsDetected', 'predictions'])
    df = df[df.ObjectsDetected != 0]
    del df['ObjectsDetected']
    df = df.sort_values(by='Segundo')
    df['Segundo'] = pd.to_numeric(df['Segundo'])
    # Apply the function to the 'seconds' column
    df['time'] = df['Segundo'].apply(seconds_to_hhmmss)
    df = df.drop(columns=['Segundo'])
    output_csv_path = 'predictions.csv'
    df.to_csv(output_csv_path, index=False)
    print(f"Predictions saved to '{output_csv_path}'")
    return output_csv_path


