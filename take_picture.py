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
    
    # Set highest resolution supported
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    if not cap.isOpened():
        messagebox.showerror("Error", "Webcam not accessible.")
        return

    # Countdown
    for i in range(3, 0, -1):
        ret, frame = cap.read()
        if not ret:
            break
        display = frame.copy()
        cv2.putText(display, str(i), (300, 300), cv2.FONT_HERSHEY_SIMPLEX, 6, (0, 0, 255), 12)
        cv2.imshow("Taking Photo", display)
        cv2.waitKey(1000)

    # Take final picture
    ret, frame = cap.read()
    if ret:
        filename = get_next_filename(save_path)
        cv2.imwrite(filename, frame)
        messagebox.showinfo("Success", f"Saved photo as {filename}")
    else:
        messagebox.showerror("Error", "Failed to capture image.")

    cap.release()
    cv2.destroyAllWindows()

# GUI
root = tk.Tk()
root.title("Take Face Photo")

tk.Label(root, text="Enter name:").pack(pady=5)
name_entry = tk.Entry(root, width=30)
name_entry.pack(pady=5)

tk.Button(root, text="Take Photo", command=take_photo).pack(pady=10)

root.mainloop()
