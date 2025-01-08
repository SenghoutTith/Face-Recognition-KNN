from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN
import time
from sklearn.preprocessing import MinMaxScaler
import heapq

def knn_train(train, test, k=5, distance_threshold=22):
    # Extract features and labels
    X_train, y_train = train[:, :-1], train[:, -1]
    
    # Calculate Euclidean distances using np.linalg.norm for efficiency
    distances = np.linalg.norm(X_train - test, axis=1)
    
    # print('d', distances)
    
    # Find top-k nearest neighbors using heapq for efficiency
    k_indices = heapq.nsmallest(k, range(len(distances)), key=lambda i: distances[i]) #heapq.nsmallest(k, iterable, key=None)
    # k: Number of smallest elements to retrieve.
    # iterable: The input collection (e.g., list, range).
    # key: (Optional) A function that computes a value for comparison.
    
    smallest_distances = [distances[i] for i in k_indices]
    
    # Get the labels of the k nearest neighbors
    nearest_labels = y_train[k_indices]
    
    # Ensure the labels are integers (if they are floats)
    nearest_labels = nearest_labels.astype(int)
    
    # Find the most frequent label (predicted class)
    predicted_label = np.bincount(nearest_labels).argmax()
    
    # print('kkk', k_indices, smallest_distances)
    # print('{} {}\n'.format(nearest_labels, np.bincount(nearest_labels).argmax()))
    
    # Get the minimum distance of the nearest neighbor
    min_distance = distances[k_indices[0]]

    # If the minimum distance is above the threshold, return 'unknown'
    if min_distance > distance_threshold:
        return 'unknown', round(min_distance, 2)

    return predicted_label, round(min_distance, 2)

def get_fps():
    global prev_time
    # Calculate and display FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    return f"FPS: {fps:.2f}"

SAVED_FILE = 'faces'
FF = cv2.FONT_HERSHEY_SIMPLEX
FZ = 0.5
FT = 1
fps = 0
prev_time = time.time()
current_image = None
class_id = 0
face_data = []
lables = []
names = {}

COLOR = {
    'YELLOW': (255, 211, 15)
}

for fx in os.listdir(SAVED_FILE):
    if fx.endswith('.npy'):
        names[class_id] = fx[:-4]
        data_item = np.load(f'{SAVED_FILE}/{fx}')
        face_data.append(data_item)
        
        # print('Label: {}\nName: {}\nFaces: {}\n'.format(class_id, names[class_id], data_item.shape[0]))

        target = class_id * np.ones((data_item.shape[0],))
        
        class_id += 1
        lables.append(target)

face_dataset = np.concatenate(face_data, axis=0)
face_lables = np.concatenate(lables, axis=0).reshape(-1, 1)

scaler = MinMaxScaler()
face_data_scaled = scaler.fit_transform(face_dataset)

train_set = np.concatenate((face_data_scaled, face_lables), axis=1)

class FaceRecognitionApp(App):
    def build(self):
        self.capture = cv2.VideoCapture(0)  # OpenCV camera capture
        self.detector = MTCNN()  # MTCNN detector

        # Main Layout
        main_layout = BoxLayout(orientation='vertical', padding=10, spacing=10)

        # Top section for name input
        name_section = BoxLayout(orientation='horizontal', size_hint=(1, 0.1))
        name_label = Label(text="Name:", size_hint=(0.2, 1), font_size=20)
        name_input = TextInput(hint_text="Please input name...", size_hint=(0.8, 1), multiline=False)
        name_section.add_widget(name_label)
        name_section.add_widget(name_input)

        # Middle section for camera preview and buttons
        middle_section = BoxLayout(orientation='horizontal', spacing=10, size_hint=(1, 0.8))

        # Camera preview area
        self.camera_layout = BoxLayout()
        self.camera_image = Image()
        self.camera_layout.add_widget(self.camera_image)
        middle_section.add_widget(self.camera_layout)

        # Buttons for Test, Training, and Collect
        button_layout = BoxLayout(orientation='vertical', spacing=10, size_hint=(0.3, 1))
        test_button = Button(text="Test", size_hint=(1, 0.3))
        train_button = Button(text="Training", size_hint=(1, 0.3))
        collect_button = Button(text="Collect", size_hint=(1, 0.3))
        button_layout.add_widget(test_button)
        button_layout.add_widget(train_button)
        button_layout.add_widget(collect_button)

        middle_section.add_widget(button_layout)

        # Bottom section for notes
        notes_section = BoxLayout(orientation='vertical', size_hint=(1, 0.2), padding=10, spacing=5)
        notes = [
            "1. Scan only 1 person at a time",
            "2. Move your head around",
            "3. Make sure the yellow box focuses on your face"
        ]
        for note in notes:
            note_label = Label(text=note, font_size=14, halign="left", valign="middle")
            note_label.bind(size=note_label.setter('text_size'))
            notes_section.add_widget(note_label)

        # Adding all sections to the main layout
        main_layout.add_widget(name_section)
        main_layout.add_widget(middle_section)
        main_layout.add_widget(notes_section)

        # Schedule camera updates
        Clock.schedule_interval(self.update_camera, 1.0 / 33.0)

        return main_layout

    def update_camera(self, dt):
        ret, frame = self.capture.read()
        if ret:
            # Do not flip the frame for processing (detection)
            # small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect faces
            results = self.detector.detect_faces(rgb)

            # Process faces if detected
            if results:
                for res in results:
                    confidence = res['confidence']
                    if confidence < 0.95:
                        continue
                    confidence = int(confidence * 100) / 100

                    x, y, w, h = res['box']
                    face = frame[y:y + h, x:x + w]
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                    face = cv2.resize(face, (100, 100))

                    current_image = face
                    face = face.reshape(1, -1)
                    face = scaler.transform(face)

                    # Predict the label and get the distance
                    predicted_label, min_distance = knn_train(train_set, face)

                    # Map the prediction to the name (if not 'unknown')
                    pred_name = predicted_label if predicted_label == 'unknown' else names[int(predicted_label)]

                    # Place text on the original frame
                    cv2.putText(frame, pred_name, (x, y - 20), FF, FZ, COLOR['YELLOW'], FT)
                    cv2.putText(frame, str(confidence), (x, y - 5), FF, FZ, (42, 219, 95), FT)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), FT)


            cv2.putText(frame, get_fps(), (10, 20), FF, FZ, (0, 255, 255), FT)
            
            # Flip only the frame for display (for the user to see)
            frame = cv2.flip(frame, 0)

            # Convert the frame to texture for display
            buf = frame.tobytes()
            texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
            texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            self.camera_image.texture = texture


    def on_stop(self):
        # Release the camera when the app is closed
        self.capture.release()

if __name__ == "__main__":
    FaceRecognitionApp().run()
