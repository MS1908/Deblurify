import numpy as np
import torch
import torch.nn as nn
import functools
import albumentations as albu
from deblur_gan.fpn_inception import FPNInception
from deblur_gan.fpn_mobilenet import FPNMobileNetV2


def get_normalize():
    normalize = albu.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    normalize = albu.Compose([normalize], additional_targets={'target': 'image'})

    def process(a, b):
        r = normalize(image=a, target=b)
        return r['image'], r['target']

    return process


class DeblurGANProcessor:

    def load_inception_model(self, incep_weight_path):
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
        model = FPNInception(norm_layer=norm_layer)
        model = nn.DataParallel(model)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load(incep_weight_path, map_location=device)['model'])
        self.inception_model = model.to(device)
        self.inception_model.train(True)

    def load_mobilenetv2_model(self, mbnetv2_weight_path, pretrained_path=None):
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=True)
        pretrained = pretrained_path is not None
        model = FPNMobileNetV2(norm_layer=norm_layer, pretrained=pretrained, pretrained_path=pretrained_path)
        model = nn.DataParallel(model)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load(mbnetv2_weight_path, map_location=device)['model'])
        self.mobilenetv2_model = model.to(device)
        self.mobilenetv2_model.train(True)

    def __init__(self, incep_weight_path, mbnetv2_weight_path, mbnet_pretrained=None):
        if incep_weight_path is not None:
            self.load_inception_model(incep_weight_path)
        else:
            self.inception_model = None
        if mbnetv2_weight_path is not None:
            self.load_mobilenetv2_model(mbnetv2_weight_path, mbnet_pretrained)
        else:
            self.mobilenetv2_model = None
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

    def deblur_quick(self, img):
        img, h, w = self._preprocess(img)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        with torch.no_grad():
            inputs = [img.to(device)]
            pred = self.mobilenetv2_model(*inputs)
        return self._postprocess(pred)[:h, :w, :]

    def __call__(self, img):
        img, h, w = self._preprocess(img)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        with torch.no_grad():
            inputs = [img.to(device)]
            pred = self.inception_model(*inputs)
        return self._postprocess(pred)[:h, :w, :]


class DeblurGeneratorSingleton:
    __deblur_instance_gan = None
    __weight_path = None

    def __new__(cls, weight_path=None):
        if cls.__deblur_instance_gan is None or cls.__weight_path != weight_path and weight_path is not None:
            cls.__deblur_instance_gan = super(DeblurGeneratorSingleton, cls).__new__(cls)
            cls.__deblur_instance_gan = DeblurGANProcessor(weight_path['incep'], weight_path['mbnetv2'],
                                                           weight_path['mbnetv2_pretrained'])
            cls.__weight_path = weight_path
        return cls.__deblur_instance_gan