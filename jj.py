# ai_pdf_image_editor.py
# AI-Powered PDF & Image Editor with Professional GUI and Seamless Text Replacement

import os
import threading
import time
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk

import cv2
import easyocr
import fitz  # PyMuPDF
import numpy as np
import pytesseract
from PIL import Image, ImageTk, ImageDraw, ImageFont
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# Configure pytesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

# Initialize OCR tools
OCR_READER = easyocr.Reader(['en'], gpu=False)
TROCR_MODEL = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")
TROCR_PROCESSOR = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TROCR_MODEL.to(DEVICE)

# ─────────────────────────────────────
# GUI SPLASH SCREEN
class SplashScreen(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.overrideredirect(True)
        self.configure(bg='black')
        w, h = 400, 300
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")
        tk.Label(self, text="AI Editor Starting...", fg="white", bg="black", font=("Helvetica", 20)).pack(expand=True)
        self.progress = ttk.Progressbar(self, length=300, mode="determinate")
        self.progress.pack(pady=20)
        threading.Thread(target=self.load).start()

    def load(self):
        for i in range(101):
            time.sleep(0.02)
            self.progress["value"] = i
            self.update_idletasks()
        self.destroy()

# ─────────────────────────────────────
# MAIN APP GUI
class AIEditorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.withdraw()
        SplashScreen(self)
        self.after(2200, self.start_gui)

    def start_gui(self):
        self.deiconify()
        self.title("AI-Powered PDF & Image Editor – SHIFTECH LTD")
        self.geometry("1280x800")
        self.configure(bg="#1f1f1f")

        self.canvas = tk.Canvas(self, bg="#1f1f1f", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.toolbar = tk.Frame(self, bg="#2a2a2a")
        self.toolbar.pack(side=tk.TOP, fill=tk.X)

        for label, cmd in [
            ("Open", self.open_file),
            ("Save", self.save_file),
            ("Prev Page", self.prev_page),
            ("Next Page", self.next_page),
            ("Zoom", self.set_zoom)
        ]:
            b = tk.Button(self.toolbar, text=label, command=cmd, bg="#444", fg="white")
            b.pack(side=tk.LEFT, padx=4, pady=4)

        self.reset_state()

    def reset_state(self):
        self.filepath = None
        self.is_pdf = False
        self.pdf = None
        self.page_idx = 0
        self.zoom = 1.0
        self.raw_img = None
        self.cv_img = None
        self.text_boxes = []

    def open_file(self):
        mode = messagebox.askquestion("Mode", "Edit a PDF? Yes = PDF, No = Image")
        path = filedialog.askopenfilename(filetypes=[("PDF or Image", "*.pdf;*.png;*.jpg;*.jpeg")])
        if not path:
            return

        self.reset_state()
        self.filepath = path
        self.is_pdf = path.endswith(".pdf")

        if self.is_pdf:
            self.pdf = fitz.open(path)
            self.page_idx = 0
            self.show_pdf_page()
        else:
            self.raw_img = Image.open(path).convert("RGB")
            self.cv_img = cv2.imread(path)
            self.show_image()

    def show_pdf_page(self):
        page = self.pdf.load_page(self.page_idx)
        pix = page.get_pixmap(matrix=fitz.Matrix(self.zoom, self.zoom))
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        self.raw_img = img
        self.cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        self.display_image()

    def show_image(self):
        img = self.raw_img.resize((int(self.raw_img.width*self.zoom), int(self.raw_img.height*self.zoom)), Image.LANCZOS)
        self.cv_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        self.display_image()

    def display_image(self):
        self.tkimg = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(self.cv_img, cv2.COLOR_BGR2RGB)))
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.tkimg)
        self.detect_text_and_overlay()

    def detect_text_and_overlay(self):
        gray = cv2.cvtColor(self.cv_img, cv2.COLOR_BGR2GRAY)
        results = OCR_READER.readtext(gray)
        for (bbox, text, prob) in results:
            (tl, tr, br, bl) = bbox
            x, y = int(tl[0]), int(tl[1])
            w, h = int(tr[0] - tl[0]), int(bl[1] - tl[1])
            self.add_textbox(x, y, w, h, text)

    def add_textbox(self, x, y, w, h, text):
        entry = tk.Entry(self.canvas, font=("Arial", max(12, int(12*self.zoom))),
                         fg="white", bg="#222", insertbackground="white", relief="flat")
        entry.insert(0, text)
        entry.place(x=x, y=y, width=w, height=h)
        self.text_boxes.append((entry, (x, y, w, h)))

    def prev_page(self):
        if self.is_pdf and self.page_idx > 0:
            self.page_idx -= 1
            self.show_pdf_page()

    def next_page(self):
        if self.is_pdf and self.page_idx < self.pdf.page_count - 1:
            self.page_idx += 1
            self.show_pdf_page()

    def set_zoom(self):
        z = simpledialog.askfloat("Zoom", "Enter zoom factor (0.5–3.0)", minvalue=0.5, maxvalue=3.0)
        if z:
            self.zoom = z
            self.show_pdf_page() if self.is_pdf else self.show_image()

    def save_file(self):
        if self.is_pdf:
            for entry, (x, y, w, h) in self.text_boxes:
                page = self.pdf.load_page(self.page_idx)
                tx, ty = x / self.zoom, y / self.zoom
                page.insert_text((tx, ty), entry.get(), fontsize=12)
            path = filedialog.asksaveasfilename(defaultextension=".pdf")
            if path:
                self.pdf.save(path)
        else:
            mask = np.zeros(self.cv_img.shape[:2], dtype=np.uint8)
            for entry, (x, y, w, h) in self.text_boxes:
                cv2.rectangle(mask, (x, y), (x+w, y+h), 255, -1)
            inpainted = cv2.inpaint(self.cv_img, mask, 3, cv2.INPAINT_TELEA)
            result = Image.fromarray(cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB))
            draw = ImageDraw.Draw(result)
            for entry, (x, y, w, h) in self.text_boxes:
                draw.text((x, y), entry.get(), fill="black", font=ImageFont.load_default())
            path = filedialog.asksaveasfilename(defaultextension=".png")
            if path:
                result.save(path)
        messagebox.showinfo("Saved", "Your file has been saved successfully.")

# ─────────────────────────────────────
if __name__ == '__main__':
    app = AIEditorApp()
    app.mainloop()
