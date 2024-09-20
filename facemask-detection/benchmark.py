import onnxruntime
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt

def preprocess_image(image):
    image = cv2.resize(image, (448, 448))  # Resize sekali saja
    image = np.array(image) / 255.0  # Skala nilai piksel
    image = image.transpose(2, 0, 1)  # Transposisi channel
    image = np.expand_dims(image, axis=0).astype(np.float32)  # Tambahkan dimensi batch
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
    
    model_input = session.get_inputs()
    original_image = cv2.imread("facemask-dataset/valid/images/maksssksksss13.png")
    
    preprocess_times = []
    inference_times = []
    postprocess_times = []

    # Loop 100 kali
    for i in range(100):
        # Preprocessing
        start_preprocess = time.perf_counter()
        image = preprocess_image(original_image)
        end_preprocess = time.perf_counter()
        preprocess_time = (end_preprocess - start_preprocess) * 1000
        preprocess_times.append(preprocess_time)
        
        # Inference
        start_inference = time.perf_counter()
        outputs = session.run(None, {model_input[0].name: image})
        end_inference = time.perf_counter()
        inference_time = (end_inference - start_inference) * 1000
        inference_times.append(inference_time)

        # Postprocessing
        start_postprocess = time.perf_counter()
        detections = postprocess(outputs)
        end_postprocess = time.perf_counter()
        postprocess_time = (end_postprocess - start_postprocess) * 1000
        postprocess_times.append(postprocess_time)
        
        print(f"Iteration {i+1}: Preprocess Time: {preprocess_time:.2f} ms, Inference Time: {inference_time:.2f} ms , Postprocess Time: {postprocess_time:.2f} ms")

    # Hitung rata-rata waktu
    avg_preprocess_time = np.mean(preprocess_times)
    avg_inference_time = np.mean(inference_times)
    avg_postprocess_time = np.mean(postprocess_times)

    print(f"\nAverage Preprocess Time: {avg_preprocess_time:.2f} ms")
    print(f"\nAverage Inference Time: {avg_inference_time:.2f} ms")
    print(f"\nAverage Postprocess Time: {avg_postprocess_time:.2f} ms")

    # Visualisasi dengan Matplotlib
    iterations = list(range(1, 101))

    plt.figure(figsize=(12, 6))

    # Plot Preprocessing Times
    plt.subplot(1, 3, 1)
    plt.plot(iterations, preprocess_times, label="Preprocessing Time", color="blue")
    plt.axhline(y=avg_preprocess_time, color='red', linestyle='--', label=f'Avg Preprocess Time: {avg_preprocess_time:.2f} ms')
    plt.xlabel("Iteration")
    plt.ylabel("Time (ms)")
    plt.title("Preprocessing Time per Iteration")
    plt.legend()

    # Plot Inference Times
    plt.subplot(1, 3, 2)
    plt.plot(iterations, inference_times, label="Inference Time", color="green")
    plt.axhline(y=avg_inference_time, color='red', linestyle='--', label=f'Avg Inference Time: {avg_inference_time:.2f} ms')
    plt.xlabel("Iteration")
    plt.ylabel("Time (ms)")
    plt.title("Inference Time per Iteration")
    plt.legend()

    # Plot Postprocess Times
    plt.subplot(1, 3, 3)
    plt.plot(iterations, postprocess_times, label="Postprocess Time", color="red")
    plt.axhline(y=avg_postprocess_time, color='red', linestyle='--', label=f'Avg Postprocess Time: {avg_postprocess_time:.2f} ms')
    plt.xlabel("Iteration")
    plt.ylabel("Time (ms)")
    plt.title("Postprocess Time per Iteration")
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()