from MyHandler import MyHandler
import numpy as np
from PIL import Image

_service = MyHandler()

def handle(data, context):
    res = []
    if not _service.initialized:
        _service.initialize(context)
    
    if data is None:
        return None

    metrics = context.metrics

    is_damaged = False

    # preprocess
    print('in preprocess')
    image, file_name = _service.preprocess_one_image(data, metrics)
    print('out preprocess')

    # GDC inference
    print('in GDC')
    result_resnet = _service.resnet_inference(image)
    print('out GDC')

    confidence_score = max(result_resnet).item()
    # conf > 0.1
    print(confidence_score)
    if confidence_score > 0.5:
        is_damaged = True

    print(is_damaged)
    print(image.shape)

    if is_damaged:
        is_damaged_num = "true"
    else:
        is_damaged_num = "false"

    # LDD inference
    if is_damaged:
        print('is_damaged')
        print('in LDD')
        effunet_data = _service.effunet_inference(image)
        print('out LDD')

        print('in postprocess')
        flag = _service.effunet_postprocess(effunet_data)
        print('in postprocess')

        if not flag:
            is_damaged_num = 'false'
            confidence_score = 0.0

        confidence_score = {'conf':confidence_score, 'is_damaged': is_damaged_num}

        res.append(confidence_score)
        return res

    else:
        print('is_nomal')
        print('in postprocess')
        _service.is_not_damaged_postprocess(image)
        print('in postprocess')

        nomal_score = {'conf':0.0, 'is_damaged': is_damaged}

        res.append(nomal_score)
        return res

    
    
    
    
