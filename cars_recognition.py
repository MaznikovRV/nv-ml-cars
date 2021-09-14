import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
import os
import random
import cv2
#from google.colab.patches import cv2_imshow

from efficientnet_pytorch import EfficientNet

from nv_ml_controller.ml_base_model import BaseMachineLearningModel
from nv_ml_controller.ml_task import BaseMlTask

class RESPONSE(BaseModel):
    status: str
    sleep_time: int

    class Config:
        title = 'RESPONSE'
        extra = 'forbid'


class REQUEST(BaseModel):
    images: List[str]

    class Config:
        title = 'REQUEST'
        extra = 'allow'


class CarsRecognizer(BaseMachineLearningModel):
    """
    Recognition of special transport in the image
    """
    model_cars_detection = 'weights/model_cars_detection'

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model_cars_detection_path = os.path.join(
            self.model_cars_detection,
            os.listdir(self.model_cars_detection)[0]
        )



        self.cars_classification_model = EfficientNet.from_pretrained('efficientnet-b0',num_classes=5)
        self.cars_classification_model.load_state_dict(torch.load(self.model_cars_detection_path, map_location='cpu'))
        self.cars_classification_model.to(self.device)
        self.cars_classification_model.eval()

        self.cars_detection_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        self.cars_detection_model.classes = [2,5,7]
        self.cars_detection_model.to(self.device)
        self.cars_detection_model.eval()

    
    def predict_label(self, img, labels,display_classes = None):
        """
        Отображает картинку и предсказывает вероятности всех классов
        :param img: картинка 
        :param labels: подписи классов
        """     
        self.tfms= transforms.Compose([transforms.Resize((224,224)),
                           transforms.ToTensor(),
                           transforms.Normalize(0.5, 0.5)])

        with torch.no_grad():
            outputs = self.cars_classification_model(self.tfms(img).unsqueeze(0).to(self.device))
        
        all_cls = len(outputs[0])
        if display_classes:
            num_cls = display_classes if display_classes<=all_cls else all_cls
        else:
            num_cls = all_cls

        idx = torch.topk(outputs, k=num_cls).indices.squeeze(0).tolist()[0]
        car_label = labels[idx]

        print("\n")

        for idx in torch.topk(outputs, k=num_cls).indices.squeeze(0).tolist():
            prob = torch.softmax(outputs, dim=1)[0, idx].item()
            print('{label:<75} ({p:.2f}%)'.format(label=labels[idx], p=prob*100))

        return car_label

    def detect_and_recognize(self, img):
        
        classes = ['ambulance', 'common', 'fire-truck', 'police', 'tractor']
        #detect object
        result = self.cars_detection_model(img)
        crop = result.crop('C:\\Users\\mazni\\Desktop\\NV_cars_recognition')
        xyxy = result.xyxy[0]

        #crop picture
        square = 0
        car_coords = ()
        largest_square = -9999

        if len(xyxy) > 1:
            for i in range(len(xyxy)):
                left = int(xyxy[i][0])
                upper = int(xyxy[i][1])
                right = int(xyxy[i][2])
                lower = int(xyxy[i][3])
                square = (right - left) * (lower - upper)

            if square >= largest_square:
                largest_square = square
                car_coords = (left, upper, right, lower)
        else:
            left = int(xyxy[0][0])
            upper = int(xyxy[0][1])
            right = int(xyxy[0][2])
            lower = int(xyxy[0][3])
            largest_square = (right - left) * (lower - upper)
            car_coords = (left, upper, right, lower)
            
        im = Image.open(img)
        im_crop = im.crop(car_coords)   

        car_label = self.predict_label(im_crop, classes)

        return car_coords, car_label
