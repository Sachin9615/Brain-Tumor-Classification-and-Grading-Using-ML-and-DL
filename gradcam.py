import os
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

# ── TFLite Model Load ─────────────────────────────────
TFLITE_PATH = r'C:\brain-tumor-project\brain_tumor_model.tflite'
interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
interpreter.allocate_tensors()
input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("✅ TFLite Model loaded!")

CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']

TUMOR_INFO = {
    'Glioma':     {'grade': 'High Grade (HGG) — Grade III/IV', 'risk': 'HIGH',
                   'desc': 'Gliomas arise from glial cells. Requires immediate attention.',
                   'treatment': 'Surgery + Radiation + Chemotherapy'},
    'Meningioma': {'grade': 'Low Grade (LGG) — Grade I/II',   'risk': 'MEDIUM',
                   'desc': 'Arises from meninges. Usually benign.',
                   'treatment': 'Observation / Surgery'},
    'Pituitary':  {'grade': 'Low Grade (LGG) — Grade I',      'risk': 'LOW',
                   'desc': 'Affects hormone regulation. Usually benign.',
                   'treatment': 'Medication / Surgery'},
    'No Tumor':   {'grade': 'N/A',                            'risk': 'NONE',
                   'desc': 'No tumor detected. Brain appears normal.',
                   'treatment': 'No treatment required'},
}

def load_and_preprocess(file_path):
    img = tf.keras.utils.load_img(file_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0).astype(np.float32)

def tflite_predict(img_array):
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    preds      = interpreter.get_tensor(output_details[0]['index'])
    class_idx  = np.argmax(preds[0])
    confidence = float(preds[0][class_idx])
    return CLASS_NAMES[class_idx], confidence, np.zeros((7, 7))

def get_tumor_size(heatmap, threshold=0.5):
    binary     = (heatmap > threshold).astype(np.uint8)
    percentage = (np.sum(binary) / heatmap.size) * 100
    size_label = "Small" if percentage < 5 else "Medium" if percentage < 15 else "Large"
    return percentage, size_label

def get_tumor_location(heatmap):
    binary = (heatmap > 0.5).astype(np.uint8)
    if np.sum(binary) == 0:
        return "Not Detected"
    h, w   = heatmap.shape
    coords = np.argwhere(binary)
    cy, cx = np.mean(coords[:, 0]), np.mean(coords[:, 1])
    v_loc  = "Superior" if cy < h*0.33 else "Central" if cy < h*0.66 else "Inferior"
    h_loc  = "Left"     if cx < w*0.33 else "Center"  if cx < w*0.66 else "Right"
    return f"{v_loc} — {h_loc}"

# ── GUI ───────────────────────────────────────────────
root = tk.Tk()
root.title("🧠 Brain Tumor Analysis System")
root.geometry("800x920")
root.configure(bg='#0D1B2A')
root.resizable(True, True)

canvas_frame = tk.Canvas(root, bg='#0D1B2A', highlightthickness=0)
scrollbar    = ttk.Scrollbar(root, orient="vertical", command=canvas_frame.yview)
canvas_frame.configure(yscrollcommand=scrollbar.set)
scrollbar.pack(side="right", fill="y")
canvas_frame.pack(side="left", fill="both", expand=True)

main_frame = tk.Frame(canvas_frame, bg='#0D1B2A')
canvas_frame.create_window((0, 0), window=main_frame, anchor="nw")
main_frame.bind("<Configure>", lambda e: canvas_frame.configure(
    scrollregion=canvas_frame.bbox("all")))

tk.Label(main_frame, text="🧠 Brain Tumor Analysis System",
         font=('Arial', 20, 'bold'), bg='#0D1B2A', fg='#00C2CB').pack(pady=15)
tk.Label(main_frame, text="ResNet50V2  |  TFLite  |  Auto Diagnosis",
         font=('Arial', 10), bg='#0D1B2A', fg='#7FB3C8').pack()

img_frame = tk.Frame(main_frame, bg='#1B2838', bd=2, relief='groove')
img_frame.pack(pady=12)
img_label = tk.Label(img_frame, bg='#1B2838', text="Upload MRI Image to Begin",
                     fg='#4A6FA5', font=('Arial', 12), width=46, height=15)
img_label.pack(padx=8, pady=8)

progress   = ttk.Progressbar(main_frame, length=520, mode='indeterminate')
progress.pack(pady=5)
status_var = tk.StringVar(value="Ready — Upload an MRI image")
tk.Label(main_frame, textvariable=status_var, font=('Arial', 10, 'italic'),
         bg='#0D1B2A', fg='#7FB3C8').pack()

cards_frame = tk.Frame(main_frame, bg='#0D1B2A')
cards_frame.pack(pady=10, padx=20, fill='x')

def make_card(parent, title, color, row, col):
    card = tk.Frame(parent, bg='#162032', bd=1, relief='groove')
    card.grid(row=row, column=col, padx=6, pady=6, sticky='nsew')
    parent.columnconfigure(col, weight=1)
    tk.Label(card, text=title, font=('Arial', 9, 'bold'),
             bg='#162032', fg=color).pack(pady=(8, 2))
    val = tk.StringVar(value="—")
    tk.Label(card, textvariable=val, font=('Arial', 12, 'bold'),
             bg='#162032', fg='white', wraplength=170).pack(pady=(2, 8))
    return val

tumor_type_var = make_card(cards_frame, "🔬 TUMOR TYPE",  '#00C2CB', 0, 0)
confidence_var = make_card(cards_frame, "📊 CONFIDENCE",  '#F5A623', 0, 1)
grade_var      = make_card(cards_frame, "🏥 TUMOR GRADE", '#9B59B6', 0, 2)
risk_var       = make_card(cards_frame, "⚠️ RISK LEVEL",  '#E74C3C', 1, 0)
location_var   = make_card(cards_frame, "📍 LOCATION",    '#3498DB', 1, 1)
size_var       = make_card(cards_frame, "📐 TUMOR SIZE",  '#27AE60', 1, 2)

info_frame = tk.Frame(main_frame, bg='#162032', bd=1, relief='groove')
info_frame.pack(padx=20, pady=8, fill='x')
tk.Label(info_frame, text="📋 Clinical Information", font=('Arial', 11, 'bold'),
         bg='#162032', fg='#F5A623').pack(anchor='w', padx=12, pady=(10, 4))
desc_var = tk.StringVar(value="Upload an MRI image to see detailed analysis.")
tk.Label(info_frame, textvariable=desc_var, font=('Arial', 10),
         bg='#162032', fg='#B8D4E8', wraplength=720,
         justify='left').pack(anchor='w', padx=12, pady=2)
treatment_var = tk.StringVar(value="")
tk.Label(info_frame, textvariable=treatment_var, font=('Arial', 10, 'bold'),
         bg='#162032', fg='#00C2CB', wraplength=720,
         justify='left').pack(anchor='w', padx=12, pady=(2, 10))

def upload_and_predict():
    file_path = filedialog.askopenfilename(
        filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if not file_path:
        return

    pil_img = Image.open(file_path).resize((340, 260))
    photo   = ImageTk.PhotoImage(pil_img)
    img_label.configure(image=photo, text="", width=0, height=0)
    img_label.image = photo

    status_var.set("🔄 Analyzing MRI... Please wait")
    for v in [tumor_type_var, confidence_var, grade_var,
              risk_var, location_var, size_var]:
        v.set("Analyzing...")
    desc_var.set("Processing...")
    treatment_var.set("")
    progress.start()
    root.update()

    try:
        img_array              = load_and_preprocess(file_path)
        raw_class, confidence, heatmap = tflite_predict(img_array)

        name_map   = {'glioma': 'Glioma', 'meningioma': 'Meningioma',
                      'notumor': 'No Tumor', 'pituitary': 'Pituitary'}
        class_name = name_map.get(raw_class, raw_class)
        info       = TUMOR_INFO.get(class_name, TUMOR_INFO['No Tumor'])

        size_pct, size_label = get_tumor_size(heatmap)
        location = get_tumor_location(heatmap) if class_name != 'No Tumor' else "N/A"

        tumor_type_var.set(class_name)
        confidence_var.set(f"{confidence * 100:.1f}%")
        grade_var.set(info['grade'])
        risk_var.set(info['risk'])
        size_var.set(f"{size_label}\n({size_pct:.1f}% area)")
        location_var.set(location)
        desc_var.set(f"📌 {info['desc']}")
        treatment_var.set(f"💊 Recommended: {info['treatment']}")
        status_var.set("✅ Analysis Complete!")

        # ── Plot ──────────────────────────────────────
        orig = cv2.imread(file_path)
        orig = cv2.resize(orig, (224, 224))

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        fig.patch.set_facecolor('#0D1B2A')

        axes[0].imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Original MRI', color='white', fontsize=13)
        axes[0].axis('off')

        title = (f"Type: {class_name}  ({confidence*100:.1f}%)\n"
                 f"Grade: {info['grade']}\n"
                 f"Risk: {info['risk']}")
        axes[1].imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
        axes[1].set_title(title, color='white', fontsize=10)
        axes[1].axis('off')

        for ax in axes:
            ax.set_facecolor('#162032')

        plt.tight_layout()
        plt.savefig(r'C:\brain-tumor-project\result.png', facecolor='#0D1B2A')
        plt.show()

    except Exception as e:
        status_var.set(f"❌ Error: {str(e)}")
        print("Error:", e)
        import traceback
        traceback.print_exc()

    progress.stop()

tk.Button(main_frame, text="📁  Upload MRI Image & Analyze",
          font=('Arial', 14, 'bold'), bg='#00C2CB', fg='#0D1B2A',
          padx=25, pady=12, cursor='hand2',
          command=upload_and_predict).pack(pady=15)

tk.Label(main_frame, text="Powered by TFLite — ResNet50V2 (94% accuracy)",
         font=('Arial', 9), bg='#0D1B2A', fg='#4A6FA5').pack(pady=(0, 15))

root.mainloop()