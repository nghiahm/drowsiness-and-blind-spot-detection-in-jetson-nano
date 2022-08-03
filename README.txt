=============================================Ngủ gật=============================================

Thu thập tập dữ liệu ảnh mắt
$ python3 collect_dataset.py --camera-index=0 --dataset-dir=data/dataset/ --number-image=240

Chia tập dữ liệu ra theo từng folder
$ python3 split_dataset.py --dataset-dir=data/dataset/ --images-dir=data/images/ --number-images=240 --number-train-image=180

Huấn luyện mạng nơ-ron CNN
$ python3 train_cnn.py --model-dir=models/cnn/model_cnn.h5 --batch-size=32 --learning-rate=0.001 --epochs=20 --validation-epochs=10

Chuyển đổi sang mô hình ONNX
$ python3 cnn_onnx_export.py --input-model=models/cnn/model_cnn.h5 --output-model=models/cnn/model_cnn.onnx

=============================================Điểm mù=============================================

Huấn luyện mạng nơ-ron SSD
$ python3 train_ssd.py --dataset-type=voc --data=data/images/ --model-dir=models/ --batch-size=4 --epochs=10

Chuyển đổi sang mô hình ONNX
$ python3 ssd_onnx_export.py --model-dir=models/

=============================================Project=============================================

Chạy project real-time
$ python3 run.py




