import cv2
import onnxruntime as ort
from datasets import raw_data
import numpy as np
import matplotlib.pyplot as plt
import random

def preprocess_image(image):
    image = np.array(image) / 255.
    image = cv2.resize(image,(224,224))
    image = image.transpose(2,0,1)
    image = np.expand_dims(image,axis=0).astype(np.float32)
    return image

def main(session):
    # load data test for inference with onnx
    _, test_data = raw_data()
    labels = ["cats","dogs"]
    random_integer = random.randint(0,len(test_data))
    test_image = test_data[random_integer][0]

    # test model with test_data index 0
    image = preprocess_image(test_image)

    # inference session
    model_input = session.get_inputs()
    pred = session.run(None, {model_input[0].name : image})
    
    # show images pred with matplotlib (you can also use cv2.imshow)
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(image.squeeze(0).transpose(1,2,0))
    plt.axis("off")
    plt.title(f"Pred : {labels[pred[0].argmax(1).item()]}")
    plt.subplot(1,2,2)
    plt.imshow(test_data[0][0])
    plt.axis("off")
    plt.title(f"Pred : {labels[pred[0].argmax(1).item()]}")
    plt.show()

if __name__ == "__main__":
    # create session onnxruntime
    session = ort.InferenceSession('./cat-and-dogs.onnx')

    # inference with opencv
    main(session)