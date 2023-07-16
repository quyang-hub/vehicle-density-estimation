# OSTNet: Overlapping Splitting Transformer Network with Integrated Density Loss for Vehicle Density Estimation

**We will upload the code and pre-training model after the article is accepted.**



## 1.Environment Setup

Python library dependencies.

```xml
-f https://download.pytorch.org/whl/cu113/torch_stable.html
torch==1.10.0+cu113
torchvision==0.11.1+cu113
torchaudio==0.10.0+cu113 
numpy>=1.16.5
scipy>=1.3.0
opencv-python
gdown
pillow
gradio
timm==0.4.12
wandb
matplotlib
```

## Data Preprocessing





**If you want to use our model quickly, you can use the test.py file directly and modify the path to the model in the file to get our results quickly.**

## Train

- train.py

```shell
python train.py
```

If you want to replace the dataset, you can use the variable dataset = 'dataset name' in the train.py file. There are also a number of parameters that can be modified to suit your needs, such as learning rate, epochs, etc.

## Test

- test.py

```shell
python test.py
```

If you want to replace the dataset, you can use the variable dataset = 'dataset name' in the test.py file.