import torch
import torchvision
from tqdm import tqdm
from datasets import download_dataset,get_dataloader
from torchvision.models.detection.faster_rcnn import fasterrcnn_resnet50_fpn,FasterRCNN_ResNet50_FPN_Weights

def main():
    # seed for reproducibility
    torch.manual_seed(42)

    # hyperparameters (tune this hyperparameter with your config)
    batch_size = 2
    learning_rate = 1e-3
    epochs = 25

    # setup dataloader
    train_loader,valid_loader,test_loader,data_classes = get_dataloader(batch_size=batch_size)

    # prepare model + fine tune
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    num_classes = len(data_classes) + 1
    input_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(input_features, num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params,lr=0.005,momentum=0.9,weight_decay=0.0005)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images,targets in tqdm(train_loader,desc=f"Training Epochs : {epoch + 1}"):
            images = list(image.to(device) for image in images)
            targets = [{ k : v.to(device) for k,v in t.items() } for t in targets]
            
            optimizer.zero_grad()
            pred = model(images,targets)
            losses = sum(loss for loss in pred.values())

            running_loss += losses.item()
            
            losses.backward()
            optimizer.step()

        # skip validation 
    
    # save model
    model.eval()
    onnx_path = "manga-text-detection.onnx"
    dummy_input = torch.randn(1, 3, 512, 512).to(device)

    torch.onnx.export(
        model,               
        dummy_input,         
        onnx_path,           
        export_params=True,  
        opset_version=20,    
        do_constant_folding=True,  
        input_names=['input'],   
        output_names=['output'], 
        dynamic_axes={'input' : {0 : 'batch_size'},'output' : {0 : 'batch_size'}})

    print(f"Model Saved : {onnx_path}")


if __name__ == "__main__":
    # Download dataset if not exists
    download_dataset()

    # Start training model function
    main()