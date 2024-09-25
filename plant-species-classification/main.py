import matplotlib.pyplot as plt
import cv2
import onnxruntime as ort
import numpy as np
from datasets import get_classes

def preprocess_image(path):
    image = cv2.imread(path)
    image = cv2.resize(image,(224,224))
    image = np.array(image) / 255.
    image = image.transpose(2,0,1)
    image = np.expand_dims(image,axis=0).astype(np.float32)
    return image

def main(session):
    labels = get_classes()
    print(labels)
    # test_image_1_path = "./data/house_plant_species/Aloe Vera/Aloe_1.jpg"
    # test_image_2_path = "./data/house_plant_species/Orchid/Orchid_3.jpg"

    # test_image_1 = preprocess_image(test_image_1_path)
    # test_image_2 = preprocess_image(test_image_2_path)

    # model_input = session.get_inputs()
    # pred_1 = session.run(None, {model_input[0].name : test_image_1})
    # pred_2 = session.run(None, {model_input[0].name : test_image_2})

    # plt.figure(figsize=(8,8))
    # plt.subplot(1,2,1)
    # plt.imshow(test_image_1.squeeze(0).transpose(1,2,0))
    # plt.axis("off")
    # plt.title(f"Pred : {labels[pred_1[0].argmax(1).item()]}")
    # plt.subplot(1,2,2)
    # plt.imshow(test_image_2.squeeze(0).transpose(1,2,0))
    # plt.axis("off")
    # plt.title(f"Pred : {labels[pred_2[0].argmax(1).item()]}")
    # plt.tight_layout()
    # plt.show()

if __name__ == "__main__":
    # create session onnxruntime
    session = ort.InferenceSession('./plant-classification.onnx',providers=ort.get_available_providers())

    # inference with opencv
    main(session)