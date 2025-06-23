from ultralytics import YOLO
import torch

print("CUDA Available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0))

model = YOLO("yolov8m.pt")
results = model.predict(source="input_videos/08fd33_4.mp4", device=0, save=True)

print(results[0])
print("==================")
for box in results[0].boxes:
    print(box)