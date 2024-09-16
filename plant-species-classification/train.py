import torch
from datasets import download_dataset,get_dataloader
from transformers import AutoImageProcessor,AutoModelForImageClassification
from tqdm import tqdm

def main(model_checkpoint):
    # seed for reproducibility
    torch.manual_seed(42)

    # hyperparameters (tune this hyperparameter with your config)
    batch_size = 32
    learning_rate = 1e-3
    epochs = 5

    # setup processor and prepare dataloader + dataset class (train and test)
    processor = AutoImageProcessor.from_pretrained(model_checkpoint,use_fast=True)
    train_loader,test_loader,dataset_class = get_dataloader(processor=processor,batch_size=batch_size)

    # prepare model google vit and freeze all layers, only train classifier
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForImageClassification.from_pretrained(model_checkpoint)
    model.classifier = torch.nn.Linear(in_features=model.classifier.in_features,out_features=len(dataset_class),bias=True)
    for params in model.parameters():
        params.requires_grad = False

    for params in model.classifier.parameters():
        params.requires_grad = True

    model.to(device)

    # training loop
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)

    for epoch in range(epochs):
        model.train()
        for image,label in tqdm(train_loader,desc=f"Training Epochs : {epoch + 1}"):
            optimizer.zero_grad()
            image,label = image.to(device),label.to(device)
            pred = model(image).logits
            loss = loss_fn(pred,label)
            loss.backward()
            optimizer.step()
            
        model.eval()
        test_loss,correct = 0,0
        with torch.no_grad():
            for image,label in tqdm(test_loader,desc=f"Validation Epochs : {epoch + 1}"):
                image,label = image.to(device),label.to(device)
                pred = model(image).logits
                test_loss += loss_fn(pred,label).item()
                correct += (pred.argmax(1) == label).sum().item()
        test_loss /= len(test_loader)
        correct /= len(test_loader.dataset)
        print(f"Validation Epoch : {epoch + 1} ========> Loss : {test_loss} | Accuracy : {correct * 100}%")

    # save model to onnx format
    model.eval()
    onnx_path = "plant-classification.onnx"
    dummy_input = torch.randn(1, 3, 224, 224).to(device)

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

    print(f"Model saved : {onnx_path}")

if __name__ == "__main__":
    # Download dataset if not exists
    download_dataset()

    # start training model function
    main(model_checkpoint="google/vit-base-patch16-224")