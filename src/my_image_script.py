from ultralytics import YOLO 

CATEGORIES = ["pedestrian", "people", "bicycle", "car", "van", "truck",
		"tricycle", "awning-tricycle", "bus", "motor"]

image_classifier = YOLO("runs/detect/train3/weights/best.pt")

answers = image_classifier.predict("004.jpg")

print(answers)
