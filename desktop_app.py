import cv2
import requests
import numpy as np
import hashlib
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk
from PIL import Image, ImageTk
import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# URL of the Flask server
add_user_url = 'http://127.0.0.1:5000/add_user'
recognize_url = 'http://127.0.0.1:5000/recognize'

# Cache to store recognized face encodings and their associated names
recognized_faces_cache = {}

class CameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition App")

        try:
            self.video_capture = cv2.VideoCapture(0)
            if not self.video_capture.isOpened():
                raise IOError("Cannot open webcam")

            self.canvas = tk.Canvas(root, width=640, height=480)
            self.canvas.pack()

            self.btn_frame = ttk.Frame(root)
            self.btn_frame.pack()

            self.add_user_btn = ttk.Button(self.btn_frame, text="Add User", command=self.prepare_to_add_user)
            self.add_user_btn.pack(side=tk.LEFT)

            self.quit_btn = ttk.Button(self.btn_frame, text="Quit", command=self.quit)
            self.quit_btn.pack(side=tk.LEFT)

            self.adding_user_mode = False
            self.num_photos = 5
            self.encodings = []

            self.face_cascade = self.load_cascade()
            self.update_frame()

        except Exception as e:
            logging.error(f"Initialization error: {e}")
            messagebox.showerror("Error", f"Initialization error: {e}")
            self.quit()

    def load_cascade(self):
        try:
            # Load the Haar Cascade from the project root
            face_cascade_path = os.path.join(os.path.dirname(__file__), 'haarcascade_frontalface_default.xml')
            logging.info(f"Loading Haar Cascade from: {face_cascade_path}")
            face_cascade = cv2.CascadeClassifier(face_cascade_path)
            if face_cascade.empty():
                raise IOError(f"Failed to load Haar Cascade classifier from {face_cascade_path}")
            return face_cascade
        except Exception as e:
            logging.error(f"Error loading Haar Cascade: {e}")
            raise

    def update_frame(self):
        try:
            ret, frame = self.video_capture.read()
            if not ret:
                self.root.after(10, self.update_frame)
                return

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray_frame,
                scaleFactor=1.1,
                minNeighbors=7,
                minSize=(100, 100),
                maxSize=(400, 400)
            )

            if self.adding_user_mode:
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    face_roi = frame[y:y + h, x:x + w]
                    _, img_encoded = cv2.imencode('.jpg', face_roi)

                    self.encodings.append(img_encoded.tobytes())
                    logging.info(f"Captured {len(self.encodings)} images")

                    if len(self.encodings) >= self.num_photos:
                        self.complete_add_user()
                        break

            else:
                for (x, y, w, h) in faces:
                    face_roi = frame[y:y + h, x:x + w]
                    _, img_encoded = cv2.imencode('.jpg', face_roi)
                    face_hash = self.hash_face_encoding(img_encoded.tobytes())

                    if face_hash in recognized_faces_cache:
                        name = recognized_faces_cache[face_hash]
                    else:
                        try:
                            response = requests.post(recognize_url, files={'file': img_encoded.tobytes()})
                            response.raise_for_status()
                            name = response.json().get('name', 'Unknown')
                            recognized_faces_cache[face_hash] = name

                            if name == "Unknown":
                                logging.info("Face not recognized. Consider adding this user.")
                        except requests.exceptions.RequestException as e:
                            logging.error(f"Error recognizing face: {e}")
                            name = "Unknown"

                    self.draw_rectangle(frame, (x, y, w, h), name)

            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

            self.root.after(10, self.update_frame)

        except Exception as e:
            logging.error(f"Error updating frame: {e}")
            messagebox.showerror("Error", f"Error updating frame: {e}")

    def prepare_to_add_user(self):
        self.adding_user_mode = True
        self.encodings = []
        messagebox.showinfo("Info", "Press 'c' to capture images of the user.")

    def complete_add_user(self):
        self.adding_user_mode = False
        user_name = simpledialog.askstring("Input", "Enter user name:", parent=self.root)

        if user_name:
            try:
                files = [('file', (f'image_{i}.jpg', enc, 'image/jpeg')) for i, enc in enumerate(self.encodings)]
                response = requests.post(add_user_url, files=files, data={'name': user_name})
                response.raise_for_status()
                logging.info(response.json())
                messagebox.showinfo("Success", f"User {user_name} added successfully!")
            except requests.exceptions.RequestException as e:
                logging.error(f"Error adding user: {e}")
                messagebox.showerror("Error", f"Error adding user: {e}")
        else:
            messagebox.showwarning("Warning", "No user name entered. User not added.")

    def draw_rectangle(self, img, bbox, name):
        x, y, w, h = bbox
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(img, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    def hash_face_encoding(self, encoding):
        return hashlib.sha256(encoding).hexdigest()

    def quit(self):
        if self.video_capture.isOpened():
            self.video_capture.release()
        self.root.quit()

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = CameraApp(root)
        root.mainloop()
    except Exception as e:
        logging.error(f"Application error: {e}")
