from flask import Flask, request, jsonify
from flask_cors import CORS
from services.deblur_utils import deblur_base64_image, DeblurProcessorSingleton, deblur_base64_image_quick

INCEP_WEIGHT_PATH = 'weights/fpn_inception.h5'
MBNET_WEIGHT_PATH = 'weight/fpn_mobilenetv2.h5'
WEIGHT_PATHS = {
    'incep': INCEP_WEIGHT_PATH,
    'mbnet': MBNET_WEIGHT_PATH
}

app = Flask(__name__)
CORS(app)


@app.route('/api/v1/deblur', methods=['GET'])
def deblur_get():
    return "Deblurify!"


@app.route('/api/v1/deblur', methods=['POST'])
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
        deblurred = deblur_base64_image_quick(b64_string=b64_image, weight_path=WEIGHT_PATHS)
        response = {'image': deblurred}
        return jsonify(response)


def load_deblur_processor():
    deblur_processor = DeblurProcessorSingleton(WEIGHT_PATHS)
    return deblur_processor


if __name__ == '__main__':
    processor = load_deblur_processor()
    app.run(host='0.0.0.0', port=8080, debug=True)
