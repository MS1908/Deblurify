from flask import Flask
from services.deblur_utils import deblur_base64_image

WEIGHT_PATH = 'weights/model_deblur.h5'

app = Flask(__name__)


@app.route('/api/v1/deblur', methods=['POST'])
def deblur_api():
    pass


app.run(host='0.0.0.0', port=8080, debug=True)
