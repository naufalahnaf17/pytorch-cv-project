import onnxruntime
import cv2
import numpy as np
import time

def preprocess_image(image):
    image = np.array(image) / 255.0
    image = image.transpose(2, 0, 1) 
    image = np.expand_dims(image, axis=0).astype(np.float32)
    return image

def postprocess(outputs):
    outputs = np.squeeze(outputs[0])
    rows = outputs.shape[0]

    boxes = []
    scores = []
    class_ids = []

    for i in range(rows):
        class_scores = outputs[i][4]
        if class_scores >= 0.8:
            x1,y1,x2,y2 = int(outputs[i][0]),int(outputs[i][1]),int(outputs[i][2]),int(outputs[i][3])
            boxes.append([x1,y1,x2,y2])

    return boxes

def main():
    # Set Inference Session dengan optimasi
    session_options = onnxruntime.SessionOptions()
    session_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = onnxruntime.InferenceSession(
        "runs/detect/Facemask-Detection/weights/best.onnx", 
        sess_options=session_options,
        providers=onnxruntime.get_available_providers()
    )
    
    # load image and input model
    model_input = session.get_inputs()
    original_image = cv2.imread("facemask-dataset/valid/images/maksssksksss4.png")
    original_image = cv2.resize(original_image,(448,448))
    original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    
    # Preprocessing
    start_preprocess = time.perf_counter()
    image = preprocess_image(original_image)
    end_preprocess = time.perf_counter()
    print(f"Preprocess Image: {(end_preprocess - start_preprocess) * 1000:.2f} ms")

    # Inference
    start_inference = time.perf_counter()
    outputs = session.run(None, {model_input[0].name: image})
    end_inference = time.perf_counter()
    print(f"Inference Time: {(end_inference - start_inference) * 1000:.2f} ms")

    # Postprocessing
    start_postprocess = time.perf_counter()
    detections = postprocess(outputs)
    end_postprocess = time.perf_counter()
    print(f"Postprocess Detection: {(end_postprocess - start_postprocess) * 1000:.2f} ms")
    for bbox in detections:
        xmin,ymin,xmax,ymax = bbox
        cv2.rectangle(original_image,(xmin,ymin),(xmax,ymax),(0, 255, 0), 2)
        cv2.putText(original_image,"classes_name",(xmin,ymin - 10),cv2.FONT_HERSHEY_SIMPLEX,0.7,(255, 0, 0),2)

    original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
    cv2.imshow("Image",original_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()