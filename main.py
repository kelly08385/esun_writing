from argparse import ArgumentParser
import base64
import time
import hashlib

import cv2
from flask import Flask
from flask import request
from flask import jsonify
import numpy as np
from collections import Counter
import logging

app = Flask(__name__)
app.config['JSON_AS_ASCII']=False

####### PUT YOUR INFORMATION HERE #######
CAPTAIN_EMAIL = 'kelly08385@gmail.com'          #
SALT = 'mynameiskelly'                        #
#########################################

from paddleocr import PaddleOCR, draw_ocr
import skimage.measure


ocr = PaddleOCR(rec_char_dict_path="./training_data_dic.txt")  # need to run only once to download and load model into memory
from opencc import OpenCC
S2T_Label = ['采', '峰', '杰', '于', '里', '梁', '游', '范', '郁', '余', '松', '岳', '群','台']

def simply_to_tradition(_str):
    return OpenCC('s2t').convert(_str)


with open('./training_data_dic.txt', 'r') as f:
    label_dict = [line.strip() for line in f]


def _FFT(img_original):
    ret3, th3 = cv2.threshold(img_original, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    dft = cv2.dft(np.float32(th3), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude_spectrum = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))

    rows, cols = img_original.shape
    crow, ccol = rows // 2, cols // 2

    # create a mask first, center square is 1, remaining all zeros
    mask = np.zeros((rows, cols, 2), np.uint8)
    mask[crow - 15:crow + 15, ccol - 15:ccol + 15] = 1

    # apply mask and inverse DFT
    fshift = dft_shift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

    cv2.normalize(img_back, img_back, 0, 255, cv2.NORM_MINMAX)
    img_back = img_back.astype(np.uint8)

    return img_back


def blur(img_original):
    img_original = img_original.astype(np.uint8)
    img_blur = cv2.GaussianBlur(img_original, (3, 3), 0)
    ret3, th3 = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th3


def generate_server_uuid(input_string):
    """ Create your own server_uuid.

    @param:
        input_string (str): information to be encoded as server_uuid
    @returns:
        server_uuid (str): your unique server_uuid
    """
    s = hashlib.sha256()
    data = (input_string + SALT).encode("utf-8")
    s.update(data)
    server_uuid = s.hexdigest()
    return server_uuid


def base64_to_binary_for_cv2(image_64_encoded):
    """ Convert base64 to numpy.ndarray for cv2.

    @param:
        image_64_encode(str): image that encoded in base64 string format.
    @returns:
        image(numpy.ndarray): an image.
    """
    img_base64_binary = image_64_encoded.encode("utf-8")
    img_binary = base64.b64decode(img_base64_binary)
    image = cv2.imdecode(np.frombuffer(img_binary, np.uint8), cv2.IMREAD_COLOR)
    return image


def predict(image):
    """ Predict your model result.

    @param:
        image (numpy.ndarray): an image.
    @returns:
        prediction (str): a word.
    """

    ####### PUT YOUR MODEL INFERENCING CODE HERE #######
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    result_arr = []

    # first
    result = ocr.ocr(image, det=False, cls=False)
    result_arr.append(list(result))

    # second
    ret3, th3 = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    result = ocr.ocr(th3, det=False, cls=False)
    result_arr.append(result)

    # third
    blur_img = blur(image_gray)
    result = ocr.ocr(blur_img, det=False, cls=False)
    result_arr.append(result)

    # forth
    # fft_img = _FFT(image)
    # result = ocr.ocr(fft_img, det=False, cls=False)
    # result_arr.append(result)

    result_arr = [item for subl in result_arr for item in subl]

    val_dict = {}
    dic_cnt = []
    val_ary = []
    max_val,max_tmp ="", -1000
    for result in result_arr:
        dic_cnt.append(result[0])
        val_ary.append(result[1])
        
        if result[1] > max_tmp:
            max_val = result[0] 
            max_tmp = result[1]
        if result[0] in val_dict:
            tmp = val_dict[result[0]]
            val_dict.update({result[0]:tmp + result[1]})
        else:
            val_dict.update({result[0]: result[1]})

    c = Counter(dic_cnt)
    max_cnt = c.most_common(1)[0][0]


    if len(max_cnt) > 1:
        max_cnt = max_cnt[0]
    if len(max_val) > 1:
        max_val = max_val[0]

    if result_arr[0][0] == result_arr[2][0]:
        ret = result_arr[0][0]
    elif np.mean(val_ary) < 5 and len(val_dict)==3 and val_dict[max_val] < 2 :
        prediction="isnull"
    elif len(val_dict)==1:
        prediction = max_cnt
    elif len(val_dict)==2:
        if (max_val==max_cnt) and (val_dict[max_cnt]/2>=3.5):
            prediction = max_val
        elif (max_val!=max_cnt) and (val_dict[max_cnt]/2>=2.5):
            prediction = max_cnt    
        elif max_val!=max_cnt and val_dict[max_val]>=2.5:
            prediction = max_val   
        else:
            prediction="isnull"    
    elif len(val_dict)==3 and val_dict[max_val] >= 2 :
        prediction = max_val
    else:
        prediction="isnull"

    if prediction=="N":
        prediction="isnull"

    
    if _check_datatype_to_string(prediction):
        return prediction



def _check_datatype_to_string(prediction):
    """ Check if your prediction is in str type or not.
        If not, then raise error.

    @param:
        prediction: your prediction
    @returns:
        True or raise TypeError.
    """
    if isinstance(prediction, str):
        return True
    raise TypeError('Prediction is not in string type.')


@app.route('/inference', methods=['POST'])
def inference():
    """ API that return your model predictions when E.SUN calls this API. """
    data = request.get_json(force=True)

    # 自行取用，可紀錄玉山呼叫的 timestamp
    esun_timestamp = data['esun_timestamp']

    # 取 image(base64 encoded) 並轉成 cv2 可用格式
    image_64_encoded = data['image']
    image = base64_to_binary_for_cv2(image_64_encoded)

    t = time.time()
    ts = str(int(t))

    cv2.imwrite("./img/img" + ts + ".png", image)
    server_uuid = generate_server_uuid(CAPTAIN_EMAIL + ts)

    try:
        answer = predict(image)
    except TypeError as type_error:
        # You can write some log...
        raise type_error
    except Exception as e:
        # You can write some log...
        raise e
    server_timestamp = int(time.time())

    log = "./bot.log"
    logging.basicConfig(filename=log,filemode="w", level=logging.DEBUG, format='%(asctime)s %(message)s',
                        datefmt='%d/%m/%Y %H:%M:%S')
    logging.info("img" + ts + ".png     answer:"+ answer)

    return jsonify({'esun_uuid': data['esun_uuid'],
                    'server_uuid': server_uuid,
                    'answer': answer,
                    'server_timestamp': server_timestamp})


if __name__ == "__main__":
    arg_parser = ArgumentParser(
        usage='Usage: python ' + __file__ + ' [--port <port>] [--help]'
    )
    arg_parser.add_argument('-p', '--port', default=8080, help='port')
    arg_parser.add_argument('-d', '--debug', default=True, help='debug')
    options = arg_parser.parse_args()

    app.run(host="0.0.0.0",debug=options.debug, port=options.port,threaded = False,processes=5)