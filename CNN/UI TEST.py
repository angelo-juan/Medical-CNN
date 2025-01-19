import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QPushButton,
                             QFileDialog, QVBoxLayout, QHBoxLayout, QMessageBox)
from PyQt5.QtGui import QPixmap, QImage
from torchvision import transforms
from PIL import Image
import os
from timm.models.resnet import ResNet  # Import the specific class(es) from timm

# Define your model class (MUST match the saved model)
class MyModel(nn.Module):  # Replace with your actual model
    def __init__(self, num_classes=1000): # Example for ResNet, adjust as needed
        super().__init__()
        self.resnet = ResNet(num_classes=num_classes) # Use the timm ResNet
    def forward(self, x):
        return self.resnet(x)


class ImageProcessor(QWidget):
    def __init__(self):
        super().__init__()

        self.num_classes = 1000 # Example for ResNet, adjust as needed
        self.model = MyModel(num_classes = self.num_classes)  # Instantiate the model

        self.model_path = "/Users/user/PythonProject/CNN/my_model.pth"
        if not os.path.exists(self.model_path):
            QMessageBox.critical(self, "Error", f"Model file not found at: {self.model_path}")
            sys.exit()

        self.model = self.load_model()
        if self.model is None:
            QMessageBox.critical(self, "Error", "Failed to load model.")
            sys.exit()

        self.initUI()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Adjust to your model's input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Standard ImageNet normalization
        ])

    def load_model(self):
        try:
            torch.serialization.add_safe_globals({'ResNet': ResNet}) # Whitelist ONLY the class you use
            self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
            self.model.eval()
            return self.model
        except FileNotFoundError:
            print(f"Error: Model file not found at {self.model_path}")
            return None
        except RuntimeError as e:
            print(f"Error loading model: {e}. Check model definition, input/output, and saving method. Check timm version.")
            return None
        except Exception as e:
            print(f"An unexpected error occurred during model loading: {e}")
            return None

    # ... (initUI, openImage, displayImage remain the same)

    def processImage(self):
        if hasattr(self, 'img') and self.model is not None:
            try:
                img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                img_tensor = self.transform(img).unsqueeze(0)

                with torch.no_grad():
                    output = self.model(img_tensor)

                if isinstance(output, tuple):
                    output = output[0]
                if len(output.shape) > 1 and output.shape[1] > 1:
                    _, predicted = torch.max(output, 1)
                    predicted_class = predicted.item()
                    self.output_label.setText(f"Prediction: {predicted_class}")
                elif len(output.shape) == 1:
                    predicted_value = output.item()
                    self.output_label.setText(f"Prediction: {predicted_value}")
                else:
                    self.output_label.setText("Prediction: Output format not handled. Check console output.")
                    print(f"Model output shape: {output.shape}")
                    print(f"Model output: {output}")

            except RuntimeError as e:
                QMessageBox.critical(self, "Error", f"PyTorch runtime error: {e}. Check input/output and preprocessing.")
                print(f"PyTorch runtime error: {e}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Processing error: {e}")
                print(f"Processing error: {e}")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = ImageProcessor()
    sys.exit(app.exec_())