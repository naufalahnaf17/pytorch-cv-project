from datasets import download_dataset,setup_datasets
from ultralytics import YOLO
import os

def main():
    os.environ['WANDB_DISABLED'] = 'true'

    # Load a model yolov10n pretrained model
    model = YOLO("yolov10n.pt") 

    # Train the model
    model.train(
        data="facemask-dataset/yolo.yaml", 
        epochs=50, 
        imgsz=448,
        batch=32,
        seed=42,
        save=True,
        device="cpu",
        name="Facemask-Detection",
    )

    # export model to onnx format
    model = YOLO("runs/detect/Facemask-Detection/weights/best.pt")
    model.export(
        format="onnx",
        imgsz=448
    )


if __name__ == "__main__":
    # Download dataset if not exists
    download_dataset()
    
    # Setup datasets VOC -> YOLO format
    setup_datasets()

    # run training
    main()