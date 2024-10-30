# Implementation of Pix2Pix
deep learning-based DIP (Pix2Pix) with PyTorch
# requirements
To install requirements:
```
python -m pip install -r requirements.txt
```

# Running

prepare dataset
```
bash download_cityscapes_dataset.sh
python cityscapes_datasets.py
```

train
```
python train.py
```

# Results
## Training
citysacpes_dataset 

image_rgb to image_semantic  

UNet

## val loss curve
![](Image/val_loss_curve.png)
## image_val during the training epochs
### epoch 50
![](Image/result_2.png)
### epoch 100
![](Image/result_2_1.png)
### epoch 150
![](Image/result_2_2.png)
### epoch 200
![](Image/result_2_3.png)
### epoch 250
![](Image/result_2_4.png)
### epoch 300
![](Image/result_2_5.png)
### epoch 400
![](Image/result_2_6.png)




