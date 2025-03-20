from ultralytics import YOLO

# Start with smaller pretrained model
model = YOLO("yolo11n-seg.pt")

# Train the model on custom dataset
results = model.train(
    data="path/to/dataset.yaml",        # path to dataset config
    epochs=100,                         # number of training epochs
    imgsz=640,                          # image size
    batch=16,                           # batch size (depends on GPU)
    patience=50,                        # early stopping patience
    save=True                           # save checkpoints
)