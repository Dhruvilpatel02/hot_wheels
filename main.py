# main.py

import os
import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from ultralytics import YOLO
from pathlib import Path
import json

# --- CONFIGURATION ---
CLASS_MODEL_PATH = r"image_classification2.pth"
DET_MODEL_PATH = r"last2.pt"
OUTPUT_DIR = r"outputs"
DEVICE = torch.device("cpu")    


class Classifier:
    """Classifies an image as 'packed' or ' '."""
    def __init__(self, model_path: str, device=DEVICE):
        self.device = device
        self.model = models.resnet18(weights=None)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 2)
        
        state_dict = torch.load(model_path, map_location=device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(device)
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.class_labels = ["packed", "unpacked"]

    def predict(self, image_path: str, threshold: float = 0.6):
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.softmax(output, dim=1)[0]
            conf, pred_idx = torch.max(probs, dim=0)

        if conf.item() < threshold:
            return "invalid", conf.item()
        return self.class_labels[pred_idx.item()], conf.item()


class Detector:
    """Detects objects in an image using a YOLO model."""
    def __init__(self, model_path: str, device=DEVICE):
        self.device = device
        self.model = YOLO(model_path)
        self.model.to(self.device)

    def run(self, image_path: str):
        results = self.model.predict(
            source=image_path,
            conf=0.25,
            iou=0.45,
            save=False,
            device=self.device.type,
            half=True if self.device.type == 'cuda' else False
        )
        result = results[0]
        detections = [result.names[int(box.cls)] for box in result.boxes]
        return detections, result

    def dump_locations(self, result, image_path: str, output_dir: str = OUTPUT_DIR):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        detections_list = []
        for box in result.boxes:
            class_name = result.names[int(box.cls)]
            x1, y1, x2, y2 = [int(coord) for coord in box.xyxy[0]]
            
            detection_info = {
                "class_name": class_name,
                "box_coordinates": {"x1": x1, "y1": y1, "x2": x2, "y2": y2}
            }
            detections_list.append(detection_info)

        image_name = Path(image_path).stem
        output_file_path = Path(output_dir) / f"{image_name}.json"

        with open(output_file_path, "w") as f:
            json.dump(detections_list, f, indent=4)
        
        return str(output_file_path)


class Validator:
    """Validates detections against classifications."""
    def __init__(self, classifier: Classifier, detector: Detector):
        self.classifier = classifier
        self.detector = detector

    def process(self, image_path: str, output_dir: str = OUTPUT_DIR):
        cls_label, cls_conf = self.classifier.predict(image_path)
        if cls_label == "invalid":
            return {"status": "rejected", "reason": f"Low classification confidence ({cls_conf:.2f})"}

        detections, result = self.detector.run(image_path)

        if not detections:
            return {"status": "rejected", "reason": "No objects detected"}
        if len(set(detections)) > 1:
            return {"status": "rejected", "reason": "Both packed and unpacked detected"}
        det_label = detections[0]
        if det_label != cls_label:
            return {"status": "rejected", "reason": f"Mismatch (classifier={cls_label}, detector={det_label})"}

        locations_file_path = self.detector.dump_locations(result, image_path, output_dir)
        
        annotated_img = result.plot()
        pred_dir = os.path.join(output_dir, "predictions")
        os.makedirs(pred_dir, exist_ok=True)
        image_name = Path(image_path).name
        save_path = os.path.join(pred_dir, f"annotated_{image_name}")
        cv2.imwrite(save_path, annotated_img)

        return {
            "status": "accepted",
            "label": det_label,
            "count": len(detections),
            "confidence": cls_conf,
            "annotated_path": save_path,
            "locations_file": locations_file_path
        }


def run_detection_and_classification(image_input: str, output_dir: str = OUTPUT_DIR):
    """
    Main function to run the full detection and classification pipeline.
    
    Args:
        image_input (str): Path to the input image.
        output_dir (str): Directory to save the output JSON labels.
        
    Returns:
        dict: A dictionary containing the processing result.
    """
    try:
        classifier = Classifier(CLASS_MODEL_PATH)
        detector = Detector(DET_MODEL_PATH)
        validator = Validator(classifier, detector)
        
        result = validator.process(image_input, output_dir)
        return result
    except Exception as e:
        print(f"An error occurred in main.py: {e}")
        return {"status": "error", "reason": str(e)}
