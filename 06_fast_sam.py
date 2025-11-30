from ultralytics import FastSAM

source = "images/image03.jpg"
model = FastSAM("models/FastSAM-s.pt")
everything_results = model(source,retina_masks=True, imgsz=1024, conf=0.8, iou=0.9)
everything_results[0].show()
