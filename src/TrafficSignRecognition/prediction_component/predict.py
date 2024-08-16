from TrafficSignRecognition.model_component.model import TrafficSignRecognitionModel, OpenCVTransformation
import torchvision.transforms as transforms
from PIL import Image
import cv2
import torch
from TrafficSignRecognition import logger
import numpy as np  

class PredictionPipeline:
    def __init__(self, img: np.ndarray):
        self.img = img
        self.best_model_parameters = torch.load('models/model.pth', map_location = torch.device('cpu'))
        self.model = TrafficSignRecognitionModel(input_shape = 1, hidden_units = 128, output_shpae = 59)
        self.transform = transforms.Compose([
            OpenCVTransformation(),
            transforms.ToPILImage(),
            transforms.ToTensor()
        ])

    def predict(self):
        logger.info('Predicting the traffic sign...')

        logger.info('loading the best model parameters...')
        self.model.load_state_dict(self.best_model_parameters['model_state_dict'])

        logger.info('Transforming the image...')
        transformed_image = self.transform(self.img)
        logger.info('Shape of the image after transformation: {}'.format(transformed_image.shape))
        logger.info('Data type of the image after transformation: {}'.format(transformed_image.dtype))

        self.model.eval()

        with torch.inference_mode():
            y_logits = self.model(transformed_image.unsqueeze(dim = 0))
            y_pred_probs = torch.softmax(y_logits, dim = 1)
            y_pred_class = torch.argmax(y_pred_probs, dim = 1).cpu().numpy()[0]

        return y_pred_probs[0][y_pred_class].cpu().numpy(), y_pred_class