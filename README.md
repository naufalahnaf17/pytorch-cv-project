# Computer Vision Project

These are some mini projects I created using Pytorch for computer vision tasks

in this repo I will create a mini project with various tasks in computer vision such as image classification, object detection, object segmentation, GAN and many more

add a star to get updates from this repo

#### Repo Update
- **17 Sep 2024 -> plant-species-classification**
- **16 Sep 2024 -> cat-and-dogs-classification**

### Dont have CUDA for Training ? Download Model Here

| Model Name         | Model URL                                                                |
| ----------------- | ------------------------------------------------------------------ |
| Cat and Dogs | [click here to download](https://drive.google.com/file/d/1Tfh30c0BShuSTTpvgeCYBHK2aZdkhjV0/view?usp=drive_link) |
| Plant Species | [click here to download](https://drive.google.com/file/d/1OYLj2TIK2oTglLJMRt2t7BW2tGLM6178/view?usp=drive_link) |

## cat-and-dogs-classification (Created : 16-09-2024)

in this project, I focused on fine-tune the resnet-50 model then exported it to onnx format.

which initially resnet-50 outputs approximately 1000 classes into just 2 classes according to the dataset that has been downloaded, namely cats and dogs

- Loss : 0.06402150790199812
- Accuracy : 98.46762234305487%
- Epochs : 5

![Pred_3](cat-and-dogs-classification/screenshot/save_3.PNG)

## plant-species-classification (Created : 17-09-2024)

in this project, I focused on fine-tune the google/vit-base-patch16-224 model then exported it to onnx format.

which initially google/vit-base-patch16-224 outputs approximately 1000 classes into just 47 classes according to the dataset that has been downloaded, namely house-plant-species

- Loss : 0.305279920695395
- Accuracy : 93.55418434246046%
- Epochs : 5

![Pred_1](plant-species-classification/screenshot/save_1.JPG)

## Todo List 
- [x] Fine tune resnet-50 on the cats and dogs dataset
  - [x] Create inference with opencv and onnxruntime (python)
  - [ ] Create inference with node.js and onnxruntime-node
  - [ ] Create Inference with kotlin
- [x] Fine tune vit-base-224 on the plant species dataset
  - [x] Create inference with opencv and onnxruntime (python)
  - [ ] Create inference with node.js and onnxruntime-node
  - [ ] Create Inference with kotlin
- [ ] Fine tune Faster-RCNN on the manga text detection dataset
  - [ ] Create inference with opencv and onnxruntime (python)
  - [ ] Create inference with node.js and onnxruntime-node
- [ ] Fine tune YOLO-V10 on the facemask dataset
  - [ ] Create inference with opencv and onnxruntime (python)
  - [ ] Create inference with node.js and onnxruntime-node
  - [ ] Create Inference with kotlin and run with input uint8


## Authors

- [@naufalahnaf17](https://www.github.com/naufalahnaf17)
