# Drowsiness and blind-spot detection in jetson nano

## 1. Drowsiness model training
### 1. Collect data
```bash
python3 collect_dataset.py --camera-index=0 --dataset-dir=data/dataset/ --number-image=240
```
### 2. Split dataset
```bash
python3 split_dataset.py --dataset-dir=data/dataset/ --images-dir=data/images/ --number-images=240 --number-train-image=180
```
### 3. Train model CNN
```bash
python3 train_cnn.py --model-dir=models/cnn/model_cnn.h5 --batch-size=32 --learning-rate=0.001 --epochs=20 --validation-epochs=10
```
### 4. Convert modal `.h5` to `.onnx`
```bash
python3 cnn_onnx_export.py --input-model=models/cnn/model_cnn.h5 --output-model=models/cnn/model_cnn.onnx
```
<br>

## 2. Blind-spot model training
### 1. Train model SSD
```bash 
python3 train_ssd.py --dataset-type=voc --data=data/images/ --model-dir=models/ --batch-size=4 --epochs=10
```
### 2. Convert modal `.h5` to `.onnx`
$ python3 ssd_onnx_export.py --model-dir=models/

<br>

## 3. Run detection
```bash
python3 run.py
```
