import numpy as np
import cv2
import io
import base64
from PIL import Image
from deblur_gan.generator_deblur_utils import DeblurGeneratorSingleton


def deblur_image(image_path, weight_path=None, save_path=None):
    deblur_processor = DeblurGeneratorSingleton(weight_path)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    deblurred = deblur_processor(image)
    deblurred = cv2.cvtColor(deblurred, cv2.COLOR_RGB2BGR)
    if save_path is not None:
        cv2.imwrite(save_path, deblurred)
    return deblurred


def b64_to_image(b64_string, save_path=None):
    if ';base64,' in b64_string:
        hdr, b64_image = b64_string.split(';base64,')
    else:
        hdr = None
        b64_image = b64_string

    b64_image = base64.b64decode(b64_image)
    image = Image.open(io.BytesIO(b64_image))
    if hdr is not None:
        image_ext = hdr.split('/')[-1]
    else:
        image_ext = 'jpg'

    if save_path is not None:
        image.save(save_path)

    return image, image_ext


def image_to_b64(image, image_ext, save_path=None):
    buffered = io.BytesIO()
    image.save(buffered, format=image_ext)
    b64_image = base64.b64encode(buffered.getvalue())
    b64_string = b64_image.decode('utf-8')
    return b64_string


def deblur_base64_image(b64_string, weight_path=None, save_path=None):
    image, image_ext = b64_to_image(b64_string)
    np_image = np.array(image.convert('RGB'))
    deblur_processor = DeblurGeneratorSingleton(weight_path)
    deblurred = deblur_processor(np_image)
    deblurred = Image.fromarray(deblurred)
    if save_path is not None:
        deblurred.save(save_path)
    deblurred_b64 = image_to_b64(deblurred, image_ext)
    return deblurred_b64


def deblur_base64_image_fast(b64_string, weight_path=None, save_path=None):
    image, image_ext = b64_to_image(b64_string)
    np_image = np.array(image.convert('RGB'))
    deblur_processor = DeblurGeneratorSingleton(weight_path)
    deblurred = deblur_processor.deblur_quick(np_image)
    deblurred = Image.fromarray(deblurred)
    if save_path is not None:
        deblurred.save(save_path)
    deblurred_b64 = image_to_b64(deblurred, image_ext)
    return deblurred_b64


def deblur_base64_image_exp(b64_string, weight_path=None, save_path=None):
    image, image_ext = b64_to_image(b64_string)
    np_image = np.array(image.convert('RGB'))
    deblur_processor = DeblurGeneratorSingleton(weight_path)
    deblurred = deblur_processor.mbnet_v3_exp(np_image)
    deblurred = Image.fromarray(deblurred)
    if save_path is not None:
        deblurred.save(save_path)
    deblurred_b64 = image_to_b64(deblurred, image_ext)
    return deblurred_b64


if __name__ == '__main__':
    # deblur_image('blur.jpg', 'deblurred.jpg')
    with open('blur_b64.txt', 'r') as fin:
        base64_image_string = fin.readline()
        b64_to_image(base64_image_string, 'blur_b64.jpg')
        weight_path = {
            'incep': '../weights/fpn_inception.h5',
            'mbnetv2': '../weights/fpn_mobilenetv2.h5',
            'mbnetv2_pretrained': '../weights/mobilenet_v2.pth.tar',
            'mbnetv3': '../weights/fpn_mobilenetv3.h5',
        }
        deblur_base64_image(base64_image_string, weight_path=weight_path, save_path='deblurred_b64.jpg')
        deblur_base64_image_fast(base64_image_string, weight_path=weight_path, save_path='deblurred_b64_fast.jpg')
