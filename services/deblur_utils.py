import numpy as np
import cv2
import torch
import torch.nn as nn
import functools
import albumentations as albu
import io
import base64
from PIL import Image
from fpn_inception import FPNInception
from mimetypes import guess_extension

MODEL_PATH = '../weights/model_deblur.h5'


def get_normalize():
    normalize = albu.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    normalize = albu.Compose([normalize], additional_targets={'target': 'image'})

    def process(a, b):
        r = normalize(image=a, target=b)
        return r['image'], r['target']

    return process


class DeblurProcessor:
    
    def __init__(self, weight_path):
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
        model = FPNInception(norm_layer=norm_layer)
        model = nn.DataParallel(model)
        model.load_state_dict(torch.load(weight_path)['model'])
        self.model = model.cuda()
        self.model.train(True)
        self.normalize_fn = get_normalize()

    @staticmethod
    def _array_to_batch(x):
        x = np.transpose(x, (2, 0, 1))
        x = np.expand_dims(x, 0)
        return torch.from_numpy(x)

    def _preprocess(self, x):
        x, _ = self.normalize_fn(x, x)

        h, w, _ = x.shape
        block_size = 32
        min_height = (h // block_size + 1) * block_size
        min_width = (w // block_size + 1) * block_size

        pad_params = {
            'mode': 'constant',
            'constant_values': 0,
            'pad_width': ((0, min_height - h), (0, min_width - w), (0, 0))
        }
        x = np.pad(x, **pad_params)

        return self._array_to_batch(x), h, w

    @staticmethod
    def _postprocess(x):
        x, = x
        x = x.detach().cpu().float().numpy()
        x = (np.transpose(x, (1, 2, 0)) + 1) / 2.0 * 255.0
        return x.astype('uint8')

    def __call__(self, img):
        img, h, w = self._preprocess(img)
        with torch.no_grad():
            inputs = [img.cuda()]
            pred = self.model(*inputs)
        return self._postprocess(pred)[:h, :w, :]


class DeblurProcessorSingleton:

    __deblur_instance = None
    __weight_path = None

    def __new__(cls, weight_path=None):
        if cls.__deblur_instance is None or cls.__weight_path != weight_path and weight_path is not None:
            cls.__deblur_instance = super(DeblurProcessorSingleton, cls).__new__(cls)
            cls.__deblur_instance = DeblurProcessor(weight_path)
            cls.__weight_path = weight_path
        return cls.__deblur_instance


def deblur_image(image_path, weight_path=None, save_path=None):
    deblur_processor = DeblurProcessorSingleton(weight_path)
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    deblurred = deblur_processor(image)
    deblurred = cv2.cvtColor(deblurred, cv2.COLOR_RGB2BGR)
    if save_path is not None:
        cv2.imwrite(save_path, deblurred)
    return deblurred


def b64_to_image(b64_string, save_path=None):
    hdr, b64_image = b64_string.split(';base64,')

    b64_image = base64.b64decode(b64_image)
    image = Image.open(io.BytesIO(b64_image))
    image_ext = guess_extension(hdr)

    if save_path is not None:
        image.save(save_path)

    return image, image_ext


def image_to_b64(image, image_ext, save_path=None):
    buffered = io.BytesIO()
    image.save(buffered, format=image_ext)
    b64_image = base64.b64encode(buffered.getvalue())
    return b64_image


def deblur_base64_image(b64_string, weight_path=None, save_path=None):
    image, image_ext = b64_to_image(b64_string)
    np_image = np.array(image.convert('RGB'))
    deblur_processor = DeblurProcessorSingleton(weight_path)
    deblurred = deblur_processor(np_image)
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
        deblur_base64_image(base64_image_string, save_path='deblurred_b64.jpg')
    
                                                                                                                        