from flask import Flask, request, jsonify
from flask_cors import CORS
from services.deblur_utils import deblur_base64_image, DeblurGeneratorSingleton, deblur_base64_image_fast, \
    deblur_base64_image_exp

INCEP_WEIGHT_PATH = 'weights/fpn_inception.h5'
MBNETV2_WEIGHT_PATH = 'weights/fpn_mobilenetv2.h5'
MBNETV2_PRETRAINED_WEIGHT_PATH = 'weights/mobilenet_v2.pth.tar'
MBNETV3_WEIGHT_PATH = 'weights/fpn_mobilenetv3.h5'
WEIGHT_PATHS = {
    'incep': INCEP_WEIGHT_PATH,
    'mbnetv2': MBNETV2_WEIGHT_PATH,
    'mbnetv2_pretrained': MBNETV2_PRETRAINED_WEIGHT_PATH,
    'mbnetv3': MBNETV3_WEIGHT_PATH
}

app = Flask(__name__)
CORS(app)


@app.route('/api/v1/deblur', methods=['GET'])
def deblur_get():
    return "Deblurify!"


@app.route('/api/v1/deblur_hd', methods=['POST'])
def deblur_api():
    if request.method == 'POST':
        data = request.get_json()
        b64_image = data['image']
        deblurred = deblur_base64_image(b64_string=b64_image, weight_path=WEIGHT_PATHS)
        response = {'image': deblurred}
        return jsonify(response)


@app.route('/api/v1/deblur_fast', methods=['POST'])
def deblur_fast_api():
    if request.method == 'POST':
        data = request.get_json()
        b64_image = data['image']
        deblurred = deblur_base64_image_fast(b64_string=b64_image, weight_path=WEIGHT_PATHS)
        response = {'image': deblurred}
        return jsonify(response)


@app.route('/api/v1/deblur_exp', methods=['POST'])
def deblur_exp_api():
    if request.method == 'POST':
        data = request.get_json()
        b64_image = data['image']
        deblurred = deblur_base64_image_exp(b64_string=b64_image, weight_path=WEIGHT_PATHS)
        response = {'image': deblurred}
        return jsonify(response)


def load_generator_deblur_processor():
    deblur_processor = DeblurGeneratorSingleton(WEIGHT_PATHS)
    return deblur_processor


if __name__ == '__main__':
    processor = load_generator_deblur_processor()
    app.run(host='0.0.0.0', port=8080, debug=True)
