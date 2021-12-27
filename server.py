from flask import Flask, request, jsonify
from services.deblur_utils import deblur_base64_image

WEIGHT_PATH = 'weights/model_deblur.h5'

app = Flask(__name__)


@app.route('/api/v1/deblur', methods=['POST'])
def deblur_api():
    if request.method == 'POST':
        data = request.get_json()
        b64_image = data['image']
        deblurred = deblur_base64_image(b64_string=b64_image, weight_path=WEIGHT_PATH)
        response = {'image': deblurred}
        return jsonify(response)


app.run(host='0.0.0.0', port=8080, debug=True)
