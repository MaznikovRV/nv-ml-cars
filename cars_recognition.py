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


class MLCarsRecognizer(BaseMachineLearningModel):
    """
    Special transport recognition
    """
    model_cars_detection = 'weights/model_cars_detection'

    def model_init(self):
        """
        Initialization for class

        :param CONFIG: config with model parameters, loaded in the BaseModel.
        """

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # get models paths
        model_cars_detection_path = os.path.join(
            self.model_cars_detection,
            os.listdir(self.model_cars_detection)[0]
        )



        self.cars_classification_model = EfficientNet.from_pretrained('efficientnet-b0',num_classes=5)
        self.cars_classification_model.load_state_dict(torch.load(model_cars_detection_path, map_location='cpu'))
        self.cars_classification_model.to(self.device)
        self.cars_classification_model.eval()

        self.cars_detection_model = torch.hub.load('ultralytics/yolov5', 'yolov5l')
        self.cars_detection_model.classes = [2,5,7]
        self.cars_detection_model.conf = 0.3
        self.cars_detection_model.to(self.device)
        self.cars_detection_model.eval()

        #self.logger.info('MLDamageClassification initialized')
    
    def _recognize_car(self, img):
        """
        Отображает картинку и предсказывает вероятности всех классов
        :param img: картинка 
        :param labels: подписи классов
        """     
        tfms= transforms.Compose([transforms.Resize((224,224)),
                           transforms.ToTensor(),
                           transforms.Normalize(0.5, 0.5)])
        
        classes = ['ambulance', 'common', 'fire-truck', 'police', 'tractor']

        with torch.no_grad():
            outputs = self.cars_classification_model(tfms(img).unsqueeze(0).to(self.device))
        
        all_cls = len(outputs[0])

        idx = torch.topk(outputs, k=all_cls).indices.squeeze(0).tolist()[0]
        car_label = classes[idx]

        #print("\n")

        #for idx in torch.topk(outputs, k=num_cls).indices.squeeze(0).tolist():
            #prob = torch.softmax(outputs, dim=1)[0, idx].item()
            #print('{label:<75} ({p:.2f}%)'.format(label=labels[idx], p=prob*100))

        return car_label

    def _get_detection_and_recognize(self, batch: list) -> Tuple[list, list]:
        
        car_coords, car_label = [], []

        for img in batch:
          #detect object
          result = self.cars_detection_model(img)
          xyxy = result.xyxy[0]

          #crop picture
          square = 0
          coords = ()
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
                      coords = (left, upper, right, lower)
                  
              im = Image.open(img)
              im_crop = im.crop(coords)
              
          elif len(xyxy) == 1:
              left = int(xyxy[0][0])
              upper = int(xyxy[0][1])
              right = int(xyxy[0][2])
              lower = int(xyxy[0][3])
              coords = (left, upper, right, lower)
              im = Image.open(img)
              im_crop = im.crop(coords)
          
          else:
              coords = []
              im_crop = Image.open(img)
        
          label = self._recognize_car(im_crop, classes)
          car_coords.append(coords)
          car_label.append(label)

        return car_coords, car_label

        
    def _get_all_predictions(self, batch: list) -> dict:

        results = []

        # get detection cars model prediction
        car_coords, car_labels = self._get_detection_and_recognize(batch)

        for label, coord in zip(car_labels, car_coords):
            
                results.append(
                        {
                            'answer': label,
                            'coords': coord
                        }
                              )
        return results

    def predict(self, task: BaseMlTask) -> list:
        """
        Head predictor. Method will make prediction for the whole batch.

        :param task: BaseMlTask
        :return: lits of dicts with answers
        """

        results = []
        print('Inside predict()')

        batch = []
        for task_image in task.images:
            batch.append(task_image.b64)
        self.logger.info(f'Batch size {len(batch)}')

        if len(batch) != 0:
            
            try:
                # convert batch to np.array format
                converted_batch = []
                for _, img_b64 in enumerate(batch):
                    image = np.array(Image.open(
                        io.BytesIO(base64.b64decode(img_b64))))
                    print(image.shape)
                    if image.shape[2] > 3:
                        image = image[:, :, 0:3]
                    print(image.shape)
                    converted_batch.append(image)
            except:
                results = [
                    {
                        'answer': 'error_1',
                        'coords': []
                    }
                ]

            # get prediction
            results = self._get_all_predictions(converted_batch)

        else:
            results = [{
                'answer': 'error_0',
                'coords': []
            }]

        return results

    def health(self):
        """
        Self check.
        """
        print('Inside health()!')
        self.logger.info('Inside health()!')
        task = BaseMlTask().create_from_directory(directory_path='tests/')
        test_results = self.predict(task)
        print(test_results)


if __name__ == "__main__":
    cars_detection_model = MLCarsRecognizer()
    cars_detection_model.serve()
    # cars_detection_model.model_init()


