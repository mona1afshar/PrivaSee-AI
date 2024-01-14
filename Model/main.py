from ultralytics import YOLO

# Create a new YOLO model from scratch
# model = YOLO('yolov8n.yaml')

# Load a pretrained YOLO model 'name.pt'
#model = YOLO('test3.pt')
model = YOLO('yolov8m.pt')
#source=0: webcam
#conf=0.6: display frame that has 60% above confidence
#results = model(source='IMG_8600.jpeg', show=True, conf=0.5)

results = model.track(source='tiktok.MP4', show=True, conf=0.4, tracker="bytetrack.yaml")
#results = model.track(source=0, show=True, conf=0.7, tracker="bytetrack.yaml")
