import matplotlib.pyplot as plt
import cv2
import onnxruntime as ort
import numpy as np
from PIL import Image, ImageDraw, ImageFont

def preprocess_image(path):
    image = cv2.imread(path)
    image = cv2.resize(image, (512, 512))
    image = np.array(image) / 255.
    image = image.transpose(2, 0, 1)
    image = np.expand_dims(image, axis=0).astype(np.float32)
    return image

def main(session):
    labels = ["Text","Text","Text"]
    test_image_1_path = "./test_image/1.jpg"
    test_image_1 = preprocess_image(test_image_1_path)

    ground_truth = Image.open(test_image_1_path).convert("RGB").resize(size=(512,512))
    prediction = Image.open(test_image_1_path).convert("RGB").resize(size=(512,512))

    model_input = session.get_inputs()
    pred_1 = session.run(None, {model_input[0].name: test_image_1})

    bboxes = pred_1[0]
    labels_pred = pred_1[1]
    confidences = pred_1[2]

    plt.figure(figsize=(8,8))

    plt.subplot(1,2,1)
    plt.imshow(ground_truth)
    plt.axis("off")
    plt.title("Ground Truth")

    plt.subplot(1,2,2)
    draw = ImageDraw.Draw(prediction)
    font = ImageFont.load_default(size=18)

    for bbox, label, confidence in zip(bboxes, labels_pred, confidences):
        if confidence >= 0.75:
            draw.rectangle(bbox.tolist(), outline="red", width=2)
            text = f"{labels[int(label)]}: {confidence:.2f}"
            text_position = (bbox[0], bbox[1] - 20) 
            draw.text(text_position, text, fill="red", font=font)
    
    plt.imshow(prediction)
    plt.axis("off")
    plt.title("Prediction")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # create session onnxruntime
    session = ort.InferenceSession('./manga-text-detection.onnx', providers=ort.get_available_providers())

    # inference with opencv
    main(session)
