import tkinter as tk
from tkinter import filedialog, messagebox
import cv2 as cv
import numpy as np
from PIL import Image, ImageTk

from sface import SFace

SFACE_MODEL_PATH = "sface_2021dec.onnx"
YUNET_MODEL_PATH = "face_detection_yunet_2023mar.onnx"

CANVAS_W = 480
CANVAS_H = 480

# -------------------------------
# Unicode-safe image read
# -------------------------------
def imread_unicode(path):
    try:
        with open(path, "rb") as f:
            data = f.read()
        return cv.imdecode(np.frombuffer(data, np.uint8), cv.IMREAD_COLOR)
    except:
        return None

# -------------------------------
# Load YuNet + SFace
# -------------------------------
detector = cv.FaceDetectorYN.create(
    YUNET_MODEL_PATH,
    "",
    (320, 320),
    score_threshold=0.6,
    nms_threshold=0.3,
    top_k=5000
)

recognizer = SFace(SFACE_MODEL_PATH, disType=0)   # cosine

# -------------------------------
# Detect best face
# -------------------------------
def detect_best_face(bgr_img):
    h, w = bgr_img.shape[:2]
    detector.setInputSize((w, h))
    ok, faces = detector.detect(bgr_img)
    if not ok or faces is None or len(faces) == 0:
        return None

    faces = faces.reshape(-1, 15)
    best = faces[np.argmax(faces[:, 4])]
    return best[:4], best  # (bbox, full face array)

# ================================================================
# GUI
# ================================================================
class FaceCompareGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition – SFace + YuNet")
        self.root.geometry("1100x700")

        self.img1 = None
        self.img2 = None

        self.img1_scale = 1
        self.img2_scale = 1
        self.img1_offset = (0, 0)
        self.img2_offset = (0, 0)

        self.crop1_box = None
        self.crop2_box = None

        self.rect1_start = None
        self.rect2_start = None
        self.rect1_id = None
        self.rect2_id = None

        # CANVASES
        self.canvas1 = tk.Canvas(root, width=CANVAS_W, height=CANVAS_H, bg="black")
        self.canvas1.grid(row=0, column=0, padx=10, pady=10)

        self.canvas2 = tk.Canvas(root, width=CANVAS_W, height=CANVAS_H, bg="black")
        self.canvas2.grid(row=0, column=1, padx=10, pady=10)

        # LOAD BUTTONS
        tk.Button(root, text="Load Image 1", command=self.load_image1).grid(row=1, column=0)
        tk.Button(root, text="Load Image 2", command=self.load_image2).grid(row=1, column=1)

        # COMPARE BUTTONS
        tk.Button(root, text="Auto Compare FULL Face", width=35,
                  command=self.compare_full).grid(row=2, column=0, pady=10)

        tk.Button(root, text="Compare Selected REGION", width=30,
                  command=self.compare_manual).grid(row=2, column=1, pady=10)

        self.result_label = tk.Label(root, text="Result: ---", font=("Arial", 16))
        self.result_label.grid(row=3, column=0, columnspan=2, pady=10)

        # Mouse bindings
        self.canvas1.bind("<Button-1>", self.c1_down)
        self.canvas1.bind("<B1-Motion>", self.c1_drag)
        self.canvas1.bind("<ButtonRelease-1>", self.c1_up)

        self.canvas2.bind("<Button-1>", self.c2_down)
        self.canvas2.bind("<B1-Motion>", self.c2_drag)
        self.canvas2.bind("<ButtonRelease-1>", self.c2_up)

    # ============================================================
    # Load Images
    # ============================================================
    def load_image1(self):
        path = filedialog.askopenfilename()
        if not path:
            return
        img = imread_unicode(path)
        if img is None:
            messagebox.showerror("Error", "Unable to load Image 1")
            return
        self.img1 = img
        self.crop1_box = None
        self.display(self.canvas1, img, is_left=True)

    def load_image2(self):
        path = filedialog.askopenfilename()
        if not path:
            return
        img = imread_unicode(path)
        if img is None:
            messagebox.showerror("Error", "Unable to load Image 2")
            return
        self.img2 = img
        self.crop2_box = None
        self.display(self.canvas2, img, is_left=False)

    # ============================================================
    # Display on canvas
    # ============================================================
    def display(self, canvas, img, is_left):
        canvas.delete("all")

        h, w = img.shape[:2]
        scale = min(CANVAS_W / w, CANVAS_H / h)
        new_w, new_h = int(scale * w), int(scale * h)

        x_off = (CANVAS_W - new_w) // 2
        y_off = (CANVAS_H - new_h) // 2

        resized = cv.resize(img, (new_w, new_h))
        bg = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
        bg[y_off:y_off + new_h, x_off:x_off + new_w] = resized

        rgb = cv.cvtColor(bg, cv.COLOR_BGR2RGB)
        tk_img = ImageTk.PhotoImage(Image.fromarray(rgb))
        canvas.image = tk_img
        canvas.create_image(0, 0, anchor="nw", image=tk_img)

        if is_left:
            self.img1_scale = scale
            self.img1_offset = (x_off, y_off)
        else:
            self.img2_scale = scale
            self.img2_offset = (x_off, y_off)

    # ============================================================
    # Canvas-to-image coordinate mapping
    # ============================================================
    def map_point(self, x, y, is_left):
        if is_left:
            img = self.img1
            scale, (ox, oy) = self.img1_scale, self.img1_offset
        else:
            img = self.img2
            scale, (ox, oy) = self.img2_scale, self.img2_offset

        if img is None:
            return None, None

        if not (ox <= x <= ox + img.shape[1] * scale):
            return None, None
        if not (oy <= y <= oy + img.shape[0] * scale):
            return None, None

        xi = int((x - ox) / scale)
        yi = int((y - oy) / scale)
        return xi, yi

    # ============================================================
    # Canvas 1 Mouse Drawing
    # ============================================================
    def c1_down(self, e):
        if self.img1 is None:
            return
        self.rect1_start = (e.x, e.y)
        if self.rect1_id:
            self.canvas1.delete(self.rect1_id)

    def c1_drag(self, e):
        if self.rect1_start is None:
            return
        if self.rect1_id:
            self.canvas1.delete(self.rect1_id)
        x0, y0 = self.rect1_start
        self.rect1_id = self.canvas1.create_rectangle(x0, y0, e.x, e.y, outline="red", width=2)

    def c1_up(self, e):
        if self.rect1_start is None:
            return
        x0, y0 = self.rect1_start
        x1, y1 = e.x, e.y
        self.rect1_start = None

        p0 = self.map_point(x0, y0, True)
        p1 = self.map_point(x1, y1, True)
        if None in p0 or None in p1:
            self.crop1_box = None
            return

        x0, y0 = p0
        x1, y1 = p1
        self.crop1_box = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

    # ============================================================
    # Canvas 2 Mouse Drawing
    # ============================================================
    def c2_down(self, e):
        if self.img2 is None:
            return
        self.rect2_start = (e.x, e.y)
        if self.rect2_id:
            self.canvas2.delete(self.rect2_id)

    def c2_drag(self, e):
        if self.rect2_start is None:
            return
        if self.rect2_id:
            self.canvas2.delete(self.rect2_id)
        x0, y0 = self.rect2_start
        self.rect2_id = self.canvas2.create_rectangle(x0, y0, e.x, e.y, outline="red", width=2)

    def c2_up(self, e):
        if self.rect2_start is None:
            return
        x0, y0 = self.rect2_start
        x1, y1 = e.x, e.y
        self.rect2_start = None

        p0 = self.map_point(x0, y0, False)
        p1 = self.map_point(x1, y1, False)
        if None in p0 or None in p1:
            self.crop2_box = None
            return

        x0, y0 = p0
        x1, y1 = p1
        self.crop2_box = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))

    # ============================================================
    # FULL FACE Compare
    # ============================================================
    def compare_full(self):

        if self.img1 is None or self.img2 is None:
            messagebox.showerror("Error", "Load both images first.")
            return

        f1 = detect_best_face(self.img1)
        f2 = detect_best_face(self.img2)

        if f1 is None or f2 is None:
            messagebox.showerror("Error", "No faces detected.")
            return

        bbox1, face1 = f1
        bbox2, face2 = f2

        score, same = recognizer.match(self.img1, face1, self.img2, face2)
        percent = score * 100
        TH = 50

        if percent >= TH:
            self.result_label.config(text=f"FULL FACE: {percent:.2f}% ✓ MATCH", fg="green")
        else:
            self.result_label.config(text=f"FULL FACE: {percent:.2f}% ✗ NOT MATCH", fg="red")

    # ============================================================
    # MANUAL REGION Compare
    # ============================================================
    def compare_manual(self):

        if self.img1 is None or self.img2 is None:
            messagebox.showerror("Error", "Load both images first.")
            return
        if self.crop1_box is None or self.crop2_box is None:
            messagebox.showerror("Error", "Select region on both images.")
            return

        x0, y0, x1, y1 = self.crop1_box
        A = self.img1[y0:y1, x0:x1]

        x2, y2, x3, y3 = self.crop2_box
        B = self.img2[y2:y3, x2:x3]

        if A.size == 0 or B.size == 0:
            messagebox.showerror("Error", "Invalid region.")
            return

        feat1 = recognizer.infer(A, bbox=None).flatten()
        feat2 = recognizer.infer(B, bbox=None).flatten()

        cos = float(np.dot(feat1, feat2) / (np.linalg.norm(feat1)*np.linalg.norm(feat2)+1e-8))
        percent = (cos + 1) / 2 * 100
        percent = np.clip(percent, 0, 100)

        self.result_label.config(text=f"MANUAL REGION: {percent:.2f}%", fg="blue")

# RUN
if __name__ == "__main__":
    root = tk.Tk()
    FaceCompareGUI(root)
    root.mainloop()
