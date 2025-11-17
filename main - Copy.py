############################################# IMPORTING ################################################
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox as mess
import tkinter.simpledialog as tsd
import cv2,os
import csv
import numpy as np
from PIL import Image, ImageTk
import pandas as pd
import datetime
import time
import threading
import queue
import base64
from io import BytesIO
from collections import defaultdict
import face_recognition
import pickle

############################################# FUNCTIONS ################################################

attendance_running = False
attendance_thread = None
attendance_queue = queue.Queue()
attendance_window = None

# Performance tracking variables
performance_metrics = {
    'total_recognitions': 0,
    'successful_recognitions': 0,
    'failed_recognitions': 0,
    'processing_times': [],
    'confidence_scores': [],
    'recognition_log': []
}

# Base64 encoded placeholder user icon
user_icon_b64 = """
iVBORw0KGgoAAAANSUhEUgAAAJAAAACQCAYAAADnRuK4AAAAAXNSR0IArs4c6QAAAARnQU1BAACx
jwv8YQUAAAAJcEhZcwAADsMAAA7DAcdvqGQAAAdASURBVHhe7d1NcpswFAVwqL//6ikpCbG3tXvr
uTqJzMok7cABZIIH2IeXp6fn5+eXj/k34/H4+fl+ftzP+znftW/y+fPnz68v399Mvt/v35+fP/+v
5x/4w+v1+n5+v79fXl4+b//V+34K4Gfz+fz5+Hj8/Pz0/u7z+bT9NIDPzWbz6/X6+vrl5eVyeXp6
er48Pb/+/s3nMZ/Pn/c/52+3s/n1vD49v/5/PZ/P+6f5vD593v+Zz+fz/v/j9fr5/fr+4e/p6fl0
Pl+eb6/X1+v1/fV6/b/f75vP27+fzd/Pz2/v9+vr9f3948u3+f18Pp+f3o6//f/8/v5y3u/3y/N5
7/fr/fP99ev19b+Xn1/Pz+9f7vcrgE/Pz6/Xy+vrzefr6/X95eXt7fl/PZ+/Pj2/f/728/N33v5m
Mv9++vry8ubz4+/+v14+n83fXj4v9/vx8s3n83d+3v4aQP/x8vbm/u3l/ePj89s7gE/Pz88/n1+v
3wD6p3d/3s7f/f/86e1dAD/7f/P3+f3+DeD/v/m7v38D6J8B9E8A/TNA/xwgf/74f3t7e3t/+/gA
8P/t7f3t/e3t/b+PzwD6p4f+6WH/dLD/OsD/nSH/zpD/Z0j/dID/OkP+nSH/z5D+6QD/dYb8O0P+
nSF90wH+6wz5d4b8O0P6pgP81xny7wz5d4b0TQf4rzPk3xny7wzpBwgQIECAAIGbBAgQIECAwE0C
BAgQIEBgJgECBAgQIDATIECAAAECMwECBAgQIDATIECAAAECswECBAgQIDCTAAECBAgQmEmAAAEC
BAjMJECAAAECBGYSIIDA4QMHDvx/8uSJAAECBP5EIK+//jr58uULAQIE/gIBAQIECBDYL0CAAAEC
BPYL/DdA/oF/DQgQIECAwN8S+DdA/oF/DQgQIECAwN8SeDdA/oF/DQgQIECAwN8S+DdA/oF/DQgQ
IECAwN8S+DdA/oF/DQgQIECAwN8S+DdA/oF/DQgQIECAwN8S+DdA/oF/DQgQIECAwN8S+DdA/oF/
DQgQIECAwN8S+DdA/oF/DQgQIECAwN8S+DdA/oF/DQgQIECAwN8S+DdA/oF/DQgQIECAwN8S+DdA
/oF/DQgQIECAwN8S+DdA/oF/DQgQIECAwN8S+DdA/oF/DQgQIECAwN8S+DdA/oF/DQgQIECAwN8S
+DdA/oF/DQgQIECAwN8S+DdA/oF/DQgQIECAwN8S+DdA/oF/DQgQIECAwN8S+DdA/oF/DQgQIECA
wN8S+DdA/oF/DQgQIECAwN8S+DdA/oF/DQgQIECAwN8S+DdA/oF/DQgQIECAwN8S+DdA/oF/DQgQ
IECAwN8S+DdA/oF/DQgQIECAwN8S+DdA/oF/DQgQIECAwN8S+DdA/oF/DQgQIECAwN8S+DdA/oF/
DQgQIECAwN8S+DdA/oF/DQgQIECAwN8S+DdA/oF/DQgQIECAwN8S+DdA/oF/DQgQIECAwN8S+DdA
/oF/DQgQIECAwN8S+DdA/oF/DQgQIECAwN8S+DdA/oF/DQgQIECAwN8S+DdA/oF/DQgQIECAwN8S
+DdA/oF/DQgQIECAwN8S+DdA/oF/DQgQIECAwN8S+DdA/oF/DQgQIECAwN8S+DdA/oF/DQgQIECA
wN8S+DdA/oF/DQgQIECAwN8S+DdA/oF/DQgQIECAwN8S+P0D28u7vB1g5cAAAAAASUVORK5CYII=
"""

def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)

def save_performance_metrics():
    """Save performance metrics to CSV"""
    assure_path_exists("Performance/")
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    filename = f"Performance/Embedding_Performance_{timestamp}.csv"

    df = pd.DataFrame(performance_metrics['recognition_log'])
    df.to_csv(filename, index=False)

    # Save summary
    summary = {
        'Total Recognitions': performance_metrics['total_recognitions'],
        'Successful': performance_metrics['successful_recognitions'],
        'Failed': performance_metrics['failed_recognitions'],
        'Success Rate': f"{(performance_metrics['successful_recognitions']/max(1, performance_metrics['total_recognitions'])*100):.2f}%",
        'Avg Processing Time': f"{np.mean(performance_metrics['processing_times']):.4f}s" if performance_metrics['processing_times'] else "N/A",
        'Avg Confidence': f"{np.mean(performance_metrics['confidence_scores']):.2f}%" if performance_metrics['confidence_scores'] else "N/A"
    }

    summary_df = pd.DataFrame([summary])
    summary_filename = f"Performance/Embedding_Summary_{timestamp}.csv"
    summary_df.to_csv(summary_filename, index=False)

    mess.showinfo("Performance Saved", f"Metrics saved to:\n{filename}\n{summary_filename}")

def tick():
    time_string = time.strftime('%H:%M:%S')
    clock.config(text=time_string)
    clock.after(200,tick)

def contact():
    mess.showinfo('Contact us', "Please contact us on : 'shubham2007kr@gmail.com' ")

def check_haarcascadefile():
    exists = os.path.isfile("haarcascade_frontalface_default.xml")
    if not exists:
        mess.showerror('Some file missing', 'haarcascade_frontalface_default.xml not found! Please add it to the directory.')
        window.destroy()

def save_pass():
    assure_path_exists("TrainingImageLabel/")
    password_file = "TrainingImageLabel/psd.txt"
    if os.path.exists(password_file):
        with open(password_file, "r") as tf:
            key = tf.read()
    else:
        master.destroy()
        new_pas = tsd.askstring('Old Password not found', 'Please enter a new password below', show='*')
        if new_pas is None:
            mess.showwarning('No Password Entered', 'Password not set! Please try again.')
        else:
            with open(password_file, "w") as tf:
                tf.write(new_pas)
            mess.showinfo('Password Registered', 'New password was registered successfully!')
        return

    op = old.get()
    newp = new.get()
    nnewp = nnew.get()
    if op == key:
        if newp == nnewp:
            with open(password_file, "w") as txf:
                txf.write(newp)
            mess.showinfo('Password Changed', 'Password changed successfully!')
            master.destroy()
        else:
            mess.showerror('Error', 'New passwords do not match. Please try again.')
    else:
        mess.showerror('Wrong Password', 'Please enter the correct old password.')

def change_pass():
    global master, old, new, nnew
    master = tk.Toplevel(window)
    master.geometry("400x160")
    master.resizable(False, False)
    master.title("Change Password")
    master.configure(background="#ffffff")
    master.grab_set()

    tk.Label(master, text='Enter Old Password', bg='white', font=('Segoe UI', 12, 'bold')).place(x=10, y=10)
    old = tk.Entry(master, width=25, fg="black", relief='solid', font=('Segoe UI', 12), show='*')
    old.place(x=180, y=10)
    tk.Label(master, text='Enter New Password', bg='white', font=('Segoe UI', 12, 'bold')).place(x=10, y=45)
    new = tk.Entry(master, width=25, fg="black", relief='solid', font=('Segoe UI', 12), show='*')
    new.place(x=180, y=45)
    tk.Label(master, text='Confirm New Password', bg='white', font=('Segoe UI', 12, 'bold')).place(x=10, y=80)
    nnew = tk.Entry(master, width=25, fg="black", relief='solid', font=('Segoe UI', 12), show='*')
    nnew.place(x=180, y=80)

    tk.Button(master, text="Cancel", command=master.destroy, fg="white", bg="#f44336", font=('Segoe UI', 10, 'bold'), width=15).place(x=220, y=120)
    tk.Button(master, text="Save", command=save_pass, fg="white", bg="#4CAF50", font=('Segoe UI', 10, 'bold'), width=15).place(x=50, y=120)

def psw():
    assure_path_exists("TrainingImageLabel/")
    password_file = "TrainingImageLabel/psd.txt"
    if not os.path.exists(password_file):
        new_pas = tsd.askstring('Setup Password', 'Please enter a new password below', show='*')
        if new_pas is None:
            mess.showwarning('No Password Entered', 'Password not set! Please try again.')
        else:
            with open(password_file, "w") as tf:
                tf.write(new_pas)
            mess.showinfo('Password Registered', 'New password was registered successfully!')
        return

    with open(password_file, "r") as tf:
        key = tf.read()

    password = tsd.askstring('Password', 'Enter Password', show='*')
    if password == key:
        TrainImages()
    elif password is not None:
        mess.showerror('Wrong Password', 'You have entered the wrong password.')

def clear():
    txt.delete(0, 'end')
    message1.configure(text="1) Take Images >>> 2) Save Profile")

def clear2():
    txt2.delete(0, 'end')

def TakeImages():
    check_haarcascadefile()
    assure_path_exists("StudentDetails/")
    assure_path_exists("TrainingImage/")
    details_path = "StudentDetails/StudentDetails.csv"
    serial = 0

    if os.path.isfile(details_path):
        try:
            df_details = pd.read_csv(details_path)
            if not df_details.empty:
                serial = df_details['SERIAL NO.'].max()
        except (pd.errors.EmptyDataError, KeyError, IndexError):
            serial = 0
    else:
        with open(details_path, 'w', newline='') as csvFile1:
            writer = csv.writer(csvFile1)
            writer.writerow(['SERIAL NO.', 'ID', 'NAME'])

    serial += 1
    Id = txt.get()
    name = txt2.get()

    if not (Id and Id.isalnum() and name and name.replace(' ', '').isalpha()):
        mess.showerror("Error", "ID must be alphanumeric and Name must be alphabetic. Both fields are required.")
        return

    cam = cv2.VideoCapture(0)
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(harcascadePath)
    sampleNum = 0
    while True:
        ret, img = cam.read()
        if not ret: break
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            sampleNum += 1
            # Save full color image for better embedding extraction
            cv2.imwrite(f"TrainingImage/{name}.{serial}.{Id}.{sampleNum}.jpg", img) # Saving full color image
            cv2.imshow('Taking Images...', img)
        if cv2.waitKey(100) & 0xFF == ord('q') or sampleNum >= 100:
            break

    cam.release()
    cv2.destroyAllWindows()
    res = f"Images Taken for ID: {Id}"
    row = [serial, Id, name]
    with open(details_path, 'a+', newline='') as csvFile:
        writer = csv.writer(csvFile)
        writer.writerow(row)
    message1.configure(text=res)
    update_member_count()

def TrainImages():
    check_haarcascadefile()
    assure_path_exists("TrainingImageLabel/")
    
    image_dir = "TrainingImage"
    faces_data, ids_data, names_data = getImagesAndEmbeddings(image_dir)
    
    if not faces_data:
        mess.showerror("Error", "No faces found to train. Please take images first.")
        return
    
    try:
        embeddings_data = {
            'embeddings': faces_data,
            'ids': ids_data,
            'names': names_data
        }
        
        with open("TrainingImageLabel/face_embeddings.pkl", "wb") as f:
            pickle.dump(embeddings_data, f)
        
        message1.configure(text="Profile Saved! Deleting temporary images...")
        
        # --- NEW: AUTOMATICALLY DELETE RAW IMAGES ---
        images_deleted = 0
        for file_name in os.listdir(image_dir):
            file_path = os.path.join(image_dir, file_name)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
                    images_deleted += 1
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
        
        mess.showinfo("Optimization Complete", f"Profile saved and {images_deleted} temporary images deleted to save space.")
        message1.configure(text="Profile Saved Successfully!")
        # --- END OF NEW CODE ---
        
        update_member_count()

    except Exception as e:
        mess.showerror("Training Error", f"Could not train model: {e}")

def getImagesAndEmbeddings(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
    embeddings = []
    ids = []
    names = []
    
    total_images = len(imagePaths)
    processed_images = 0
    
    for imagePath in imagePaths:
        processed_images += 1
        # Update status message
        message1.configure(text=f"Processing image {processed_images}/{total_images}...")
        window.update_idletasks() # Force GUI update
        
        try:
            image = face_recognition.load_image_file(imagePath)
            face_encodings = face_recognition.face_encodings(image)
            
            if len(face_encodings) > 0:
                encoding = face_encodings[0]
                filename = os.path.split(imagePath)[-1]
                parts = filename.split(".")
                
                if len(parts) >= 4:
                    name = parts[0]
                    serial_no = int(parts[1])
                    
                    embeddings.append(encoding)
                    ids.append(serial_no)
                    names.append(name)
        except Exception as e:
            print(f"Error processing {imagePath}: {e}")
    
    return embeddings, ids, names

def get_attendance_status(user_id, login_status, current_time):
    last_user_status = login_status.get(user_id, {'status': 'Logout', 'time': '1970-01-01 00:00:00'})
    try:
        last_time = datetime.datetime.strptime(last_user_status['time'], '%Y-%m-%d %H:%M:%S')
    except ValueError:
        last_time = datetime.datetime.now() - datetime.timedelta(days=1)
    
    time_since_last_event = current_time - last_time

    if last_user_status['status'] == 'Login':
        if time_since_last_event.total_seconds() > 30: # 30-second cooldown for logout
            return 'Logout'
        return "Currently Logged In"
    else: # If status is Logout
        if time_since_last_event.total_seconds() > 30: # 30-second cooldown for login
            return 'Login'
        return None # In cooldown period, do nothing

class AttendanceWindow(tk.Toplevel):
    def __init__(self, on_closing_callback, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.title("Face Attendance - Embedding")
        self.geometry("1000x600")
        self.configure(bg="#2c3e50")
        self.protocol("WM_DELETE_WINDOW", on_closing_callback)

        header_frame = tk.Frame(self, bg="#34495e")
        header_frame.pack(fill="x")
        tk.Label(header_frame, text="Real-Time Attendance (Embedding)", font=("Helvetica", 22, "bold"), bg="#34495e", fg="#ecf0f1").pack(pady=10)
        main_frame = tk.Frame(self, bg="#2c3e50")
        main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        self.video_label = tk.Label(main_frame, bg="black", relief="solid", borderwidth=2)
        self.video_label.pack(side="left", expand=True, fill="both", padx=(0, 20))
        card_frame = tk.Frame(main_frame, bg="#34495e", width=300, relief="raised", bd=2)
        card_frame.pack(side="right", fill="y")
        card_frame.pack_propagate(False)
        self.greeting_label = tk.Label(card_frame, text="Scanning...", font=("Helvetica", 18, "bold"), bg="#34495e", fg="#1abc9c")
        self.greeting_label.pack(pady=(20, 10))
        self.profile_canvas = tk.Canvas(card_frame, width=150, height=150, bg="#34495e", highlightthickness=0)
        self.profile_canvas.pack(pady=10)
        self.profile_canvas.create_oval(5, 5, 145, 145, outline="#1abc9c", width=4)
        
        try:
            image_data = base64.b64decode(user_icon_b64)
            image = Image.open(BytesIO(image_data)).resize((140, 140), Image.LANCZOS)
            self.profile_photo = ImageTk.PhotoImage(image)
            self.profile_canvas.create_image(75, 75, image=self.profile_photo)
        except Exception as e:
            print(f"Could not load profile icon: {e}")
        
        self.status_label = tk.Label(card_frame, text="Status: Searching", font=("Helvetica", 16, "italic"), bg="#34495e", fg="#bdc3c7")
        self.status_label.pack(pady=10)
        self.name_label = tk.Label(card_frame, text="", font=("Helvetica", 16, "bold"), bg="#34495e", fg="#ecf0f1")
        self.name_label.pack(pady=5)
        self.id_label = tk.Label(card_frame, text="", font=("Helvetica", 14), bg="#34495e", fg="#ecf0f1")
        self.id_label.pack(pady=5)
        
        # Performance labels
        self.confidence_label = tk.Label(card_frame, text="Confidence: --", font=("Helvetica", 12), bg="#34495e", fg="#f39c12")
        self.confidence_label.pack(pady=5)
        self.processing_label = tk.Label(card_frame, text="Processing: --", font=("Helvetica", 12), bg="#34495e", fg="#9b59b6")
        self.processing_label.pack(pady=5)

        stop_button = ttk.Button(card_frame, text="Stop Camera", command=on_closing_callback, style="Stop.TButton")
        stop_button.pack(side="bottom", pady=20, padx=20, fill="x")

    def update_info(self, name, user_id, status, confidence=None, processing_time=None):
        if status == "Login":
            status_color = "#2ecc71"
        elif status == "Logout":
            status_color = "#e74c3c"
        else:
            status_color = "#f39c12"
        self.status_label.config(text=f"Status: {status}", fg=status_color)
        self.name_label.config(text=f"Name: {name}")
        self.id_label.config(text=f"ID: {user_id}")
        
        if confidence is not None:
            self.confidence_label.config(text=f"Confidence: {confidence:.1f}%")
        if processing_time is not None:
            self.processing_label.config(text=f"Processing: {processing_time*1000:.1f}ms")

    def show_searching(self):
        self.status_label.config(text="Status: Searching...", fg="#bdc3c7")
        self.name_label.config(text="")
        self.id_label.config(text="")
        self.confidence_label.config(text="Confidence: --")
        self.processing_label.config(text="Processing: --")

    def show_unknown(self):
        self.status_label.config(text="Status: Unknown User", fg="#e74c3c")
        self.name_label.config(text="Not Registered")
        self.id_label.config(text="")

def TrackImages():
    global attendance_running, attendance_window, login_status
    
    check_haarcascadefile()
    assure_path_exists("Attendance/")
    
    try:
        with open("TrainingImageLabel/face_embeddings.pkl", "rb") as f:
            embeddings_data = pickle.load(f)
        
        known_embeddings = embeddings_data['embeddings']
        known_ids = embeddings_data['ids']
        known_names = embeddings_data['names']
        
        df_students = pd.read_csv("StudentDetails/StudentDetails.csv")
    except Exception as e:
        attendance_queue.put({'action': 'popup', 'type': 'error', 'message': f"Data Missing: {e}"})
        attendance_queue.put({'action': 'stop'})
        return

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        attendance_queue.put({'action': 'popup', 'type': 'error', 'message': "Could not open camera."})
        attendance_queue.put({'action': 'stop'})
        return

    def on_closing():
        global attendance_running
        attendance_running = False
    
    attendance_window = AttendanceWindow(on_closing_callback=on_closing)
    
    def video_loop():
        if not attendance_running:
            cam.release()
            if 'attendance_window' in globals() and attendance_window and attendance_window.winfo_exists():
                attendance_window.destroy()
            globals()['attendance_window'] = None
            attendance_queue.put({'action': 'reset_button'})
            return
            
        ret, im = cam.read()
        if not ret:
            window.after(10, video_loop)
            return
        
        # Start timing
        start_time = time.time()
        
        # Resize frame for faster processing
        small_frame = cv2.resize(im, (0, 0), fx=0.5, fy=0.5)
        rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        face_locations = face_recognition.face_locations(rgb_frame, model="hog")
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        face_detected_in_frame = False
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            face_detected_in_frame = True
            
            # Scale back up face locations
            top *= 2; right *= 2; bottom *= 2; left *= 2
            cv2.rectangle(im, (left, top), (right, bottom), (255, 165, 0), 2)
            
            matches = face_recognition.compare_faces(known_embeddings, face_encoding, tolerance=0.6)
            face_distances = face_recognition.face_distance(known_embeddings, face_encoding)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                
                if matches[best_match_index] and face_distances[best_match_index] < 0.6:
                    serial = known_ids[best_match_index]
                    name = known_names[best_match_index]
                    
                    # Calculate confidence
                    confidence = max(0, (1 - face_distances[best_match_index] / 0.6)) * 100
                    
                    ts = time.time()
                    date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
                    
                    user_info = df_students.loc[df_students['SERIAL NO.'] == serial]
                    if user_info.empty:
                        performance_metrics['failed_recognitions'] += 1
                        performance_metrics['total_recognitions'] += 1
                        attendance_queue.put({'action': 'show_unknown'})
                        continue
                    
                    user_id = user_info['ID'].values[0]
                    
                    # Track performance
                    performance_metrics['successful_recognitions'] += 1
                    performance_metrics['total_recognitions'] += 1
                    performance_metrics['processing_times'].append(processing_time)
                    performance_metrics['confidence_scores'].append(confidence)
                    performance_metrics['recognition_log'].append({
                        'Timestamp': datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'),
                        'Name': name,
                        'ID': user_id,
                        'Confidence': f"{confidence:.1f}%",
                        'Processing_Time_ms': processing_time * 1000,
                        'Method': 'Embedding'
                    })
                    
                    cv2.putText(im, f"{name} ({confidence:.0f}%)", (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)
                    
                    status = get_attendance_status(user_id, login_status, datetime.datetime.fromtimestamp(ts))
                    
                    if status in ['Login', 'Logout']:
                        timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                        login_status[user_id] = {'time': datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S'), 'status': status}
                        
                        fileName = f"Attendance/Attendance_{date}.csv"
                        fieldnames = ['ID', 'Name', 'Date', 'Login Time', 'Logout Time']
                        
                        if status == 'Login':
                            new_record = {'ID': user_id, 'Name': name, 'Date': date, 'Login Time': timeStamp, 'Logout Time': ''}
                            write_header = not os.path.isfile(fileName) or os.path.getsize(fileName) == 0
                            with open(fileName, 'a', newline='') as f:
                                writer = csv.DictWriter(f, fieldnames=fieldnames)
                                if write_header: writer.writeheader()
                                writer.writerow(new_record)
                        elif status == 'Logout':
                            try:
                                df_att = pd.read_csv(fileName)
                                df_att['ID'] = df_att['ID'].astype(str)
                                mask = (df_att['ID'] == str(user_id)) & (df_att['Login Time'].notna()) & (df_att['Logout Time'].isna() | (df_att['Logout Time'] == ''))
                                if mask.any():
                                    last_login_idx = df_att[mask].index[-1]
                                    df_att.loc[last_login_idx, 'Logout Time'] = timeStamp
                                    df_att.to_csv(fileName, index=False)
                                else:
                                    new_record = {'ID': user_id, 'Name': name, 'Date': date, 'Login Time': '', 'Logout Time': timeStamp}
                                    pd.concat([df_att, pd.DataFrame([new_record])]).to_csv(fileName, index=False)
                            except (FileNotFoundError, pd.errors.EmptyDataError):
                                new_record = {'ID': user_id, 'Name': name, 'Date': date, 'Login Time': '', 'Logout Time': timeStamp}
                                with open(fileName, 'w', newline='') as f:
                                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                                    writer.writeheader()
                                    writer.writerow(new_record)
                        
                        popup_message = f"Attendance Recorded!\n\nName: {name}\nStatus: {status}\nTime: {timeStamp}\nConfidence: {confidence:.1f}%"
                        attendance_queue.put({'action': 'popup', 'type': 'info', 'message': popup_message})
                    
                    display_status = status if status else login_status.get(user_id, {}).get('status', 'Logged Out')
                    if display_status == "Login": display_status = "Currently Logged In"
                    attendance_queue.put({'action': 'update_info', 'name': name, 'user_id': user_id, 'status': display_status, 'confidence': confidence, 'processing_time': processing_time})
                else:
                    performance_metrics['failed_recognitions'] += 1
                    performance_metrics['total_recognitions'] += 1
                    attendance_queue.put({'action': 'show_unknown'})
            else:
                performance_metrics['failed_recognitions'] += 1
                performance_metrics['total_recognitions'] += 1
                attendance_queue.put({'action': 'show_unknown'})
        
        if not face_detected_in_frame:
            attendance_queue.put({'action': 'show_searching'})
        
        if 'attendance_window' in globals() and attendance_window and attendance_window.winfo_exists():
            img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            attendance_window.video_label.imgtk = imgtk
            attendance_window.video_label.configure(image=imgtk)
        
        update_performance_display()
        
        window.after(10, video_loop)
    
    login_status = {}
    video_loop()

# ... (The rest of your GUI code remains exactly the same)
# (process_queue, update_treeview, update_member_count, etc.)
# ...
######################################## GUI FRONT-END ###########################################

window = tk.Tk()
window.geometry("1280x720")
window.title("Smart Face Recognition Attendance System - Embedding")
window.configure(bg="#2c3e50")

style = ttk.Style(window)
style.theme_use("clam")

BG_COLOR = "#2c3e50"
FRAME_COLOR = "#34495e"
TEXT_COLOR = "#ecf0f1"
TITLE_COLOR_1 = "#1abc9c"
HEADER_COLOR = "#1abc9c"
BUTTON_COLOR_REG = "#3498db"
BUTTON_HOVER_REG = "#2980b9"
START_COLOR = "#27ae60"
START_HOVER = "#2ecc71"
STOP_COLOR = "#c0392b"
STOP_HOVER = "#e74c3c"
TREE_BG = "#34495e"
TREE_FIELD_BG = "#2c3e50"
TREE_HEADING_BG = "#2c3e50"
TREE_HEADING_FG = "#1abc9c"
SELECTED_COLOR = "#16a085"

FONT_TITLE = ("Roboto", 28, "bold")
FONT_HEADER = ("Roboto", 18, "bold")
FONT_LABEL = ("Roboto", 14)
FONT_BUTTON = ("Roboto", 12, "bold")
FONT_TABLE_HEADER = ("Roboto", 14, "bold")
FONT_TABLE_ROW = ("Roboto", 12)

style.configure("TNotebook", background=BG_COLOR, borderwidth=0)
style.configure("TNotebook.Tab", background="#2c3e50", foreground="#bdc3c7", font=FONT_LABEL, padding=[25, 10], borderwidth=0)
style.map("TNotebook.Tab", background=[("selected", FRAME_COLOR)], foreground=[("selected", TITLE_COLOR_1)])
style.configure("TFrame", background=FRAME_COLOR)
style.configure("TLabel", background=FRAME_COLOR, foreground=TEXT_COLOR, font=FONT_LABEL)
style.configure("Header.TLabel", font=FONT_HEADER, foreground=HEADER_COLOR)

style.configure("Reg.TButton", background=BUTTON_COLOR_REG, foreground="white", font=FONT_BUTTON, padding=12, borderwidth=0, relief="flat")
style.map("Reg.TButton", background=[("active", BUTTON_HOVER_REG)])
style.configure("Start.TButton", background=START_COLOR, foreground="white", font=FONT_BUTTON, padding=12, borderwidth=0)
style.map("Start.TButton", background=[("active", START_HOVER)])
style.configure("Stop.TButton", background=STOP_COLOR, foreground="white", font=FONT_BUTTON, padding=12, borderwidth=0)
style.map("Stop.TButton", background=[("active", STOP_HOVER)])

style.configure("Treeview", background=TREE_BG, foreground=TEXT_COLOR, fieldbackground=TREE_FIELD_BG, font=FONT_TABLE_ROW, rowheight=35, borderwidth=0)
style.map("Treeview", background=[("selected", SELECTED_COLOR)])
style.configure("Treeview.Heading", font=FONT_TABLE_HEADER, background=TREE_HEADING_BG, foreground=TREE_HEADING_FG, borderwidth=0)

top_bar = tk.Frame(window, bg=BG_COLOR, pady=15)
top_bar.pack(side="top", fill="x")

title_label = tk.Label(top_bar, text="Smart Attendance System (Embedding)", font=FONT_TITLE, bg=BG_COLOR)
title_label.pack()
title_colors = [TITLE_COLOR_1, "#27ae60", "#2980b9", "#8e44ad", "#f39c12"]
title_color_index = 0
def animate_title():
    global title_color_index
    title_color_index = (title_color_index + 1) % len(title_colors)
    title_label.config(fg=title_colors[title_color_index])
    window.after(2000, animate_title)

clock = tk.Label(window, font=("Roboto", 16, "bold"), bg=BG_COLOR, fg=TEXT_COLOR)
clock.place(relx=0.98, rely=0.03, anchor="ne")
tick()

notebook = ttk.Notebook(window)
notebook.pack(expand=True, fill="both", padx=25, pady=(10, 0))

tab_attendance = ttk.Frame(notebook, padding=25)
notebook.add(tab_attendance, text='Attendance')
ttk.Label(tab_attendance, text="Today's Attendance Log", style="Header.TLabel").pack(anchor="w", pady=(0, 20))
tv_frame = ttk.Frame(tab_attendance)
tv_frame.pack(expand=True, fill="both")
tv = ttk.Treeview(tv_frame, columns=('id', 'name', 'date', 'login', 'logout'), show='headings')
scrollbar = ttk.Scrollbar(tv_frame, orient="vertical", command=tv.yview)
tv.configure(yscrollcommand=scrollbar.set)
tv.pack(side="left", expand=True, fill="both")
scrollbar.pack(side="right", fill="y")
tv.heading('id', text='ID'); tv.column('id', width=100, anchor="center")
tv.heading('name', text='Name'); tv.column('name', width=180)
tv.heading('date', text='Date'); tv.column('date', width=120, anchor="center")
tv.heading('login', text='Login Time'); tv.column('login', width=120, anchor="center")
tv.heading('logout', text='Logout Time'); tv.column('logout', width=150, anchor="center")
# This function should be defined or available
# update_treeview() 

tab_registration = ttk.Frame(notebook, padding=25)
notebook.add(tab_registration, text='Registration')
ttk.Label(tab_registration, text="New User Registration", style="Header.TLabel").pack(anchor="w", pady=(0, 20))
ttk.Label(tab_registration, text="Enter ID:").pack(anchor="w", pady=(15, 5))
txt = ttk.Entry(tab_registration, width=40, font=FONT_LABEL)
txt.pack(fill="x", ipady=8)
ttk.Label(tab_registration, text="Enter Name:").pack(anchor="w", pady=(15, 5))
txt2 = ttk.Entry(tab_registration, width=40, font=FONT_LABEL)
txt2.pack(fill="x", ipady=8)
message1 = ttk.Label(tab_registration, text="1) Take Images >>> 2) Save Profile", foreground="#95a5a6", font=("Roboto", 11, "italic"))
message1.pack(pady=25)
btn_frame_reg = ttk.Frame(tab_registration)
btn_frame_reg.pack(fill="x", pady=10)
btn_frame_reg.columnconfigure((0, 1, 2), weight=1)
ttk.Button(btn_frame_reg, text="Take Images", command=TakeImages, style="Reg.TButton").grid(row=0, column=0, sticky="ew", padx=5)
ttk.Button(btn_frame_reg, text="Save Profile", command=psw, style="Reg.TButton").grid(row=0, column=1, sticky="ew", padx=5)
ttk.Button(btn_frame_reg, text="Clear Fields", command=lambda: [clear(), clear2()], style="Reg.TButton").grid(row=0, column=2, sticky="ew", padx=5)
message = ttk.Label(tab_registration, text="", font=("Roboto", 12))
message.pack(side="bottom", pady=10)
# This function should be defined or available
# update_member_count()

def update_treeview():
    for item in tv.get_children():
        tv.delete(item)
    date = datetime.datetime.now().strftime('%d-%m-%Y')
    fileName = f"Attendance/Attendance_{date}.csv"
    if os.path.isfile(fileName):
        try:
            df = pd.read_csv(fileName).fillna("N/A")
            for index, row in df.iterrows():
                tv.insert("", 0, values=(row['ID'], row['Name'], row['Date'], row.get('Login Time', 'N/A'), row.get('Logout Time', 'N/A')))
        except (pd.errors.EmptyDataError, FileNotFoundError):
            pass

def update_member_count():
    path = "StudentDetails/StudentDetails.csv"
    count = 0
    if os.path.isfile(path):
        try:
            df = pd.read_csv(path)
            count = len(df)
        except pd.errors.EmptyDataError: pass
    message.config(text=f"Total Registered Members: {count}")
    
def update_performance_display():
    """Update performance metrics on the dashboard"""
    total = performance_metrics['total_recognitions']
    success = performance_metrics['successful_recognitions']
    failed = performance_metrics['failed_recognitions']
    
    if total > 0:
        accuracy = (success / total) * 100
        perf_accuracy_label.config(text=f"Recognition Accuracy: {accuracy:.2f}%")
        perf_total_label.config(text=f"Total Recognitions: {total}")
        perf_success_label.config(text=f"Successful: {success}")
        perf_failed_label.config(text=f"Failed: {failed}")
        
        if performance_metrics['processing_times']:
            avg_time = np.mean(performance_metrics['processing_times']) * 1000
            perf_time_label.config(text=f"Avg Processing Time: {avg_time:.2f}ms")
        
        if performance_metrics['confidence_scores']:
            avg_conf = np.mean(performance_metrics['confidence_scores'])
            perf_conf_label.config(text=f"Avg Confidence: {avg_conf:.1f}%")

def export_to_excel():
    attendance_folder = "Attendance"
    if not os.path.isdir(attendance_folder):
        mess.showinfo("No Data", "Attendance folder not found.")
        return
    
    all_files = [os.path.join(attendance_folder, f) for f in os.listdir(attendance_folder) if f.startswith("Attendance_") and f.endswith(".csv")]
    if not all_files:
        mess.showinfo("No Data", "No attendance records found.")
        return

    df_list = []
    for file in all_files:
        try:
            df = pd.read_csv(file)
            if not df.empty:
                df_list.append(df)
        except pd.errors.EmptyDataError:
            continue

    if not df_list:
        mess.showinfo("No Data", "Attendance files are empty.")
        return

    combined_df = pd.concat(df_list, ignore_index=True)
    
    output_path = os.path.join(attendance_folder, "Complete_Attendance_Report.xlsx")
    try:
        combined_df.to_excel(output_path, index=False, engine='openpyxl')
        mess.showinfo("Export Successful", f"All attendance data exported to:\n{os.path.abspath(output_path)}")
    except Exception as e:
        mess.showerror("Export Failed", f"Could not save Excel file. Error: {e}")

def delete_user():
    del_id = entry_del_id.get().strip()
    if not del_id:
        mess.showerror("Error", "Please enter a user ID to delete.")
        return
    
    details_path = "StudentDetails/StudentDetails.csv"
    if not os.path.isfile(details_path):
        mess.showerror("Error", "StudentDetails.csv not found.")
        return
    try:
        df = pd.read_csv(details_path)
        df['ID'] = df['ID'].astype(str)
        if del_id in df['ID'].values:
            df = df[df['ID'] != del_id]
            df.to_csv(details_path, index=False)
            mess.showinfo("Success", f"User with ID {del_id} deleted from details file.")
            
            img_folder = "TrainingImage"
            # This part is now less relevant as images are deleted, but good for cleanup if process is interrupted
            removed_images = 0
            if os.path.isdir(img_folder):
                for filename in os.listdir(img_folder):
                    try:
                        file_id = filename.split('.')[2]
                        if file_id == del_id:
                            os.remove(os.path.join(img_folder, filename))
                            removed_images += 1
                    except IndexError:
                        continue
            if removed_images > 0:
                mess.showinfo("Images Deleted", f"Deleted {removed_images} remaining images for ID {del_id}.")
            
            # Re-train with remaining users, if any
            TrainImages() 
            update_member_count()
        else:
            mess.showerror("Error", f"User with ID {del_id} not found.")
    except (pd.errors.EmptyDataError, FileNotFoundError):
        mess.showerror("Error", "StudentDetails.csv is empty or not found.")

def toggle_attendance():
    global attendance_running, attendance_thread
    if attendance_running:
        attendance_running = False
    else:
        if attendance_thread and attendance_thread.is_alive():
            mess.showwarning("In Progress", "Attendance system is already running.")
            return
        attendance_running = True
        attendance_button.config(text="■ Stop Attendance", style="Stop.TButton")
        attendance_thread = threading.Thread(target=TrackImages, daemon=True)
        attendance_thread.start()

def process_queue():
    try:
        item = attendance_queue.get_nowait()
        action = item.get('action')
        
        if action == 'popup':
            if item.get('type') == 'error':
                now = time.time()
                if not hasattr(process_queue, "last_unknown_popup_time") or (now - getattr(process_queue, "last_unknown_popup_time", 0)) > 5:
                    mess.showerror("Error", item['message'])
                    process_queue.last_unknown_popup_time = now
            else:
                mess.showinfo("Attendance Success", item['message'])
            window.after(100, update_treeview)
        elif action == 'stop':
            global attendance_running
            attendance_running = False
        elif action == 'reset_button':
            attendance_button.config(text="▶ Start Attendance", style="Start.TButton")
        
        if 'attendance_window' in globals() and attendance_window and attendance_window.winfo_exists():
            if action == 'update_info':
                attendance_window.update_info(item['name'], item['user_id'], item['status'], 
                                            item.get('confidence'), item.get('processing_time'))
            elif action == 'show_searching':
                attendance_window.show_searching()
            elif action == 'show_unknown':
                attendance_window.show_unknown()
    except queue.Empty:
        pass
    finally:
        window.after(100, process_queue)


tab_delete = ttk.Frame(notebook, padding=25)
notebook.add(tab_delete, text='Delete User')
label_del = ttk.Label(tab_delete, text="Enter ID to delete user details and saved embeddings", style="Header.TLabel")
label_del.pack(anchor="w", pady=(0, 20))
entry_del_id = ttk.Entry(tab_delete, width=40, font=FONT_LABEL)
entry_del_id.pack(fill="x", ipady=8)
btn_delete = ttk.Button(tab_delete, text="Delete User", command=delete_user, style="Stop.TButton")
btn_delete.pack(pady=20)


tab_performance = ttk.Frame(notebook, padding=25)
notebook.add(tab_performance, text='Performance')
ttk.Label(tab_performance, text="Performance Metrics (Embedding Method)", style="Header.TLabel").pack(anchor="w", pady=(0, 20))

perf_frame = ttk.Frame(tab_performance)
perf_frame.pack(fill="both", expand=True)

perf_accuracy_label = ttk.Label(perf_frame, text="Recognition Accuracy: 0%", font=("Roboto", 16, "bold"), foreground="#1abc9c")
perf_accuracy_label.pack(pady=10)

perf_total_label = ttk.Label(perf_frame, text="Total Recognitions: 0", font=("Roboto", 14))
perf_total_label.pack(pady=5)

perf_success_label = ttk.Label(perf_frame, text="Successful: 0", font=("Roboto", 14), foreground="#27ae60")
perf_success_label.pack(pady=5)

perf_failed_label = ttk.Label(perf_frame, text="Failed: 0", font=("Roboto", 14), foreground="#e74c3c")
perf_failed_label.pack(pady=5)

perf_time_label = ttk.Label(perf_frame, text="Avg Processing Time: --", font=("Roboto", 14))
perf_time_label.pack(pady=5)

perf_conf_label = ttk.Label(perf_frame, text="Avg Confidence: --", font=("Roboto", 14))
perf_conf_label.pack(pady=5)

ttk.Button(perf_frame, text="Export Performance Report", command=save_performance_metrics, style="Reg.TButton").pack(pady=20)

bottom_bar = tk.Frame(window, bg=BG_COLOR, pady=20)
bottom_bar.pack(side="bottom", fill="x")
attendance_button = ttk.Button(bottom_bar, text="▶ Start Attendance", command=toggle_attendance, style="Start.TButton")
attendance_button.pack(side="left", padx=(30, 10))
ttk.Button(bottom_bar, text="Exit Application", command=window.destroy, style="Stop.TButton").pack(side="right", padx=30)

menubar = tk.Menu(window, bg=FRAME_COLOR, fg=TEXT_COLOR, activebackground=SELECTED_COLOR, activeforeground=TEXT_COLOR)
file_menu = tk.Menu(menubar, tearoff=0, bg=FRAME_COLOR, fg=TEXT_COLOR)
file_menu.add_command(label='Export to Excel', command=export_to_excel)
file_menu.add_separator()
file_menu.add_command(label='Exit', command=window.destroy)
menubar.add_cascade(label='File', menu=file_menu)
help_menu = tk.Menu(menubar, tearoff=0, bg=FRAME_COLOR, fg=TEXT_COLOR)
help_menu.add_command(label='Change Password', command=change_pass)
help_menu.add_command(label='Contact Us', command=contact)
menubar.add_cascade(label='Help', menu=help_menu)
window.config(menu=menubar)

update_treeview()
update_member_count()
process_queue()
animate_title()
window.mainloop()

