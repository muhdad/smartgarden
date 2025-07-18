import numpy as np
import tensorflow as tf
tflite = tf.lite
from PIL import Image
import json
import os
import cv2

label_info = {
    "belum_matang": {
        "description": "Kulit dan tangkai buah masih berwarna hijau, kulit lunak, dan belum siap panen.",
        "solution": "Tunggu beberapa hari hingga warna kulit menguning dan tekstur mengeras."
    },
    "setengah_matang": {
        "description": "Buah mulai menguning, masih terlihat garis vertikal warna hijau, tangkai berubah warna menjadi cokelat",
        "solution": "Biarkan beberapa hari lagi untuk pematangan sempurna atau panen jika dibutuhkan segera."
    },
    "matang": {
        "description": "Buah sudah berwarna kuning/oranye cerah, Tangkai buah telah mengering berwarna kecoklatan, teksturnya mengeras seperti gabus, dan siap dipanen.",
        "solution": "Segera panen dan simpan di tempat sejuk agar tidak cepat membusuk."
    }
}

def load_model():
    model_path = "model/labu_model.tflite"
    label_map_path = "model/label_map.json"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model TFLite tidak ditemukan: {model_path}")
    
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    if os.path.exists(label_map_path):
        with open(label_map_path, "r") as f:
            class_names = json.load(f)
    else:
        class_names = list(label_info.keys())

    return interpreter, class_names

def preprocess_image_for_prediction(image_file, target_size=224):
    try:
        image = Image.open(image_file).convert("RGB")
        image = np.array(image)

        h, w = image.shape[:2]
        if h > w:
            new_h, new_w = target_size, int(w * target_size / h)
        else:
            new_h, new_w = int(h * target_size / w), target_size

        image = cv2.resize(image, (new_w, new_h))
        delta_w = target_size - new_w
        delta_h = target_size - new_h
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        image = cv2.copyMakeBorder(image, top, bottom, left, right,
                                   cv2.BORDER_CONSTANT, value=[0, 0, 0])

        image = cv2.convertScaleAbs(image, alpha=1.2, beta=10)
        image = image.astype(np.float32)
        image = np.expand_dims(image, axis=0)
        return image

    except Exception as e:
        raise ValueError(f"Gambar tidak valid: {str(e)}")

def classify_image(image_file, threshold=0.98):
    try:
        interpreter, class_names = load_model()
        img_array = preprocess_image_for_prediction(image_file)

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        interpreter.set_tensor(input_details[0]['index'], img_array)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])

        confidence = float(np.max(predictions))
        class_idx = int(np.argmax(predictions))
        label = class_names[class_idx]

        if confidence < threshold:
            return {
                "status": "invalid",
                "message": (
                    f"❌ Confidence terlalu rendah ({confidence:.2f}). "
                    "Gambar kemungkinan bukan labu butternut."
                )
            }

        if label not in label_info:
            return {
                "status": "invalid",
                "message": f"Label '{label}' tidak dikenali."
            }

        return {
            "status": "ok",
            "label": label,
            "confidence": confidence,
            "description": label_info[label]["description"],
            "solution": label_info[label]["solution"]
        }

    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
