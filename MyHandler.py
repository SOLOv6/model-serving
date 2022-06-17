import logging
import torch
import torch.nn.functional as F
import io
from PIL import Image
from torchvision import transforms
from ts.torch_handler.base_handler import BaseHandler
from efficientunet_model import *
from gdc_model import ResnetGradCam
import os
import sys
from crfseg import CRF
import base64
import numpy as np
from tempfile import TemporaryFile
from google.cloud import storage
import cv2

def image_plus_mask(image, pred):
    # distiribution class
    dent_pred = pred[:,:,0]
    scratch_pred = pred[:,:,1]
    spacing_pred = pred[:,:,2]

    # dent, scratch, spacing mask
    dent_mask = cv2.cvtColor(dent_pred,cv2.COLOR_GRAY2RGB)
    scratch_mask = cv2.cvtColor(scratch_pred,cv2.COLOR_GRAY2RGB)
    spacing_mask = cv2.cvtColor(spacing_pred,cv2.COLOR_GRAY2RGB)

    # mask to color
    dent_mask[:,:,0] = 0 
    scratch_mask[:,:,1] = 0 
    spacing_mask[:,:,2] = 0 

    # image plus mask
    dent_image = cv2.addWeighted(image, 1, dent_mask, 0.65, 0)
    scratch_image = cv2.addWeighted(image, 1, scratch_mask, 0.65, 0)
    spacing_image = cv2.addWeighted(image, 1, spacing_mask, 0.65, 0)

    # ndarray_to_image
    dent_image=Image.fromarray(dent_image)
    scratch_image=Image.fromarray(scratch_image)
    spacing_image=Image.fromarray(spacing_image)
        
    return dent_image, scratch_image, spacing_image

class MyHandler(BaseHandler):
    """
    Custom handler for pytorch serve. This handler supports batch requests.
    For a deep description of all method check out the doc:
    https://pytorch.org/serve/custom_service.html
    """

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.transform = transforms.Compose([
            transforms.Resize((448,448)),
            transforms.ToTensor(),
        ])

    # model_pth_path = /home/model-server/tmp/models/335f81ca38ae4e5ba21787015c965928/b4-v3.pth
    # model_dir = /home/model-server/tmp/models/335f81ca38ae4e5ba21787015c965928
    def initialize(self, context):
        self.manifest = context.manifest
        self.device = torch.device("cpu")

        properties = context.system_properties
        model_dir = properties.get("model_dir")

        self.account_key_dir = os.path.join(model_dir, 'key.json')

        # Read model serialize/pth file 
        effunetb4_pth_path = os.path.join(model_dir, 'effunetb4.pt')
        if not os.path.isfile(effunetb4_pth_path):
            raise RuntimeError("Missing the model.pt file")

        resnet152_pth_path = os.path.join(model_dir, 'resnet152.pt')
        if not os.path.isfile(resnet152_pth_path):
            raise RuntimeError("Missing the model.pt file")

        # LDD(efficientunet)
        self.effunet = get_efficientunet_b4(out_channels=3, concat_input=True, pretrained=True)
        checkpoint = torch.load(effunetb4_pth_path, map_location=torch.device('cpu'))
        self.effunet.load_state_dict(checkpoint["state_dict"])
        self.effunet.to(self.device)
        self.effunet.eval()

        # GDC(resnet152)
        self.resnet = torch.jit.load(resnet152_pth_path, map_location=torch.device('cpu'))
        self.resnet.to(self.device)
        self.resnet.eval()

        self.initialized = True

    def preprocess_one_image(self, req, metrics):
        image = req[0].get('data')
        file_name = req[0].get('path')
        self.file_name = file_name.decode()

        # metrics
        metrics.add_size("SizeOfImage", len(image) / 1024, None, "kB")  
        
        img_data = base64.b64decode(image)
        dataBytesIO = io.BytesIO(img_data)
        self.image = Image.open(dataBytesIO).convert("RGB")
        self.image_numpy = np.array(self.image)

        self.image_tensor = self.transform(self.image)
        self.pre_image = self.image_tensor.unsqueeze(0)

        return self.pre_image, self.file_name

    def resnet_inference(self, x):
        preds = self.resnet.forward(x) # out -> (batch ,3)
        preds = torch.sigmoid(preds)
        preds = preds.squeeze()
        return preds
    
    def effunet_inference(self, x):
        outs = self.effunet.forward(x)
        
        return outs

    def effunet_postprocess(self, preds):   
        # crf 
        crf = CRF(n_spatial_dims=2, requires_grad=False)
        pred = crf(preds)

        # transpose & to_numpy
        pred = F.interpolate(pred, size=[self.image_numpy.shape[0],self.image_numpy.shape[1]])
        # transforms.Resize((IMAGE_HEIGHT,IMAGE_WIDTH), torchvision.transforms.InterpolationMode.NEAREST)
        
        pred = pred.squeeze()
        pred = torch.permute(pred, (1,2,0)).contiguous()
        pred = pred.detach().numpy()

        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0

        flag = True 
        if np.sum(pred) == 0.0:
            flag = False

        pred *= 255
        pred = pred.astype(np.uint8)

        print(self.image_numpy.shape)
        print(pred.shape)

        dent_image, scratch_image, spacing_image = image_plus_mask(self.image_numpy, pred)

        masks = {
        'dent':dent_image,
        'scratch': scratch_image,
        'spacing': spacing_image
        }

        # store class mask
        BUCKET_NAME = 'solov6-test-storage'
        image_name = self.file_name
        
        storage_client = storage.Client.from_service_account_json(self.account_key_dir)
        bucket = storage_client.bucket(BUCKET_NAME)

        for class_name in masks:
            destination_blob_name = f"path_inference_{class_name}/{image_name}"

            with TemporaryFile() as gcs_image:
                masks[class_name].save(gcs_image, "jpeg")
                #cv2.imwrite('gcs_image.jpg', image)
                gcs_image.seek(0)
                blob = bucket.blob(destination_blob_name)
                blob.upload_from_file(gcs_image)
                print(f"File {image_name}.png uploaded to {destination_blob_name}.")
        
        return flag

    def is_not_damaged_postprocess(self, image):
        # store class mask
        BUCKET_NAME = 'solov6-test-storage'
        image_name = self.file_name
        
        storage_client = storage.Client.from_service_account_json(self.account_key_dir)
        bucket = storage_client.bucket(BUCKET_NAME)

        classes = ['dent', 'scratch', 'spacing']

        for class_name in classes:
            destination_blob_name = f"path_inference_{class_name}/{image_name}"

            with TemporaryFile() as gcs_image:
                self.image.save(gcs_image, "jpeg")
                #cv2.imwrite('gcs_image.jpeg', gcs_image)
                gcs_image.seek(0)
                blob = bucket.blob(destination_blob_name)
                blob.upload_from_file(gcs_image)
                print(f"File {image_name}.png uploaded to {destination_blob_name}.")
    

        



