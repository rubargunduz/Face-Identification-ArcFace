import cv2
import os
import tkinter as tk
from tkinter import messagebox

# Create images directory if it doesn't exist
if not os.path.exists("images"):
    os.makedirs("images")

def get_next_filename(path):
    existing_files = [f for f in os.listdir(path) if f.endswith(".jpg") and f.split(".")[0].isdigit()]
    existing_ids = [int(f.split(".")[0]) for f in existing_files]
    next_id = max(existing_ids) + 1 if existing_ids else 1
    return os.path.join(path, f"{next_id}.jpg")


def take_photo():
    name = name_entry.get().strip()
    if not name:
        messagebox.showerror("Error", "Please enter a name.")
        return

    save_path = os.path.join("images", name)
    os.makedirs(save_path, exist_ok=True)

    # Open webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    if not cap.isOpened():
        messagebox.showerror("Error", "Webcam not accessible.")
        return

    def capture_and_save():
        ret, frame = cap.read()
        if ret:
            filename = get_next_filename(save_path)
            cv2.imwrite(filename, frame)
            messagebox.showinfo("Success", f"Saved photo as {filename}")
        else:
            messagebox.showerror("Error", "Failed to capture image.")

    # Create a window to show the live feed and a button to capture
    cv2.namedWindow("Taking Photo")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Taking Photo", frame)
        # Wait for button press in GUI
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        root.update()
        if getattr(root, 'capture_flag', False):
            capture_and_save()
            root.capture_flag = False

    cap.release()
    cv2.destroyAllWindows()

# GUI
root = tk.Tk()
root.title("Take Face Photo")

tk.Label(root, text="Enter name:").pack(pady=5)
name_entry = tk.Entry(root, width=30)
name_entry.pack(pady=5)

def on_capture():
    root.capture_flag = True

tk.Button(root, text="Start Camera & Take Photos", command=take_photo).pack(pady=10)
tk.Button(root, text="Capture Photo", command=on_capture).pack(pady=10)

root.capture_flag = False
root.mainloop()
