from ultralytics import YOLO

# Load a model
def load_train():

	model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

	# Train the model
	results = model.train(data="VisDrone.yaml", epochs=1, imgsz=640)
	return model, results

if __name__ == "__main__":
	model, results = load_train()
