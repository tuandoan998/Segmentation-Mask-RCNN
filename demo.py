import os
import time
import datetime
import logging
import flask
from flask import request, send_from_directory
import werkzeug
import tornado.wsgi
import tornado.httpserver
import numpy as np
from PIL import Image, ImageDraw
import skimage
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib import patches,  lines
from matplotlib.patches import Polygon
import io
import sys

from deepfashion2 import DeepFashion2Config
# Import Mask RCNN
Mask_RCNN_DIR = os.path.abspath("Mask_RCNN")
sys.path.append(Mask_RCNN_DIR)  # To find local version of the library
from mrcnn import visualize, utils
from mrcnn import model as modellib


ALLOWED_IMAGE_EXTENSIONS = set(['png', 'bmp', 'jpg', 'jpe', 'jpeg', 'gif', 'tif', 'tiff'])

# Obtain the flask app object
app = flask.Flask(__name__)
REPO_DIRNAME = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = 'TMP'


def allowed_file(filename):
    return (
        '.' in filename and
        filename.rsplit('.', 1)[1] in ALLOWED_IMAGE_EXTENSIONS
    )

def draw_predict(image, image_name, boxes, masks, class_ids, class_names,
                figsize=(12, 12), show_mask=True):
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(image)

    # Generate random colors
    colors = visualize.random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                            alpha=0.7, linestyle="dashed",
                            edgecolor=colors[i], facecolor='none')
        ax.add_patch(p)

        # Label
        class_id = class_ids[i]
        label = class_names[class_id]
        caption = label
        ax.text(x1, y1 + 8, caption, color='w', size=11, backgroundcolor="k")

        # Mask
        mask = masks[:, :, i]
        masked_image = visualize.apply_mask(masked_image, mask, colors[i])

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=colors[i])
            ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8))
    fig.savefig(os.path.join(UPLOAD_FOLDER, 'predict_'+image_name), bbox_inches='tight', pad_inches=0)



@app.route('/')
def index():
    return flask.render_template('index.html', has_result=False)

# Dectect image from upload
@app.route('/upload', methods=['POST'])
def upload():
    for upload in request.files.getlist("imagefile"):
        global img_name, img_path
        img_name = upload.filename
        img_path = "/".join([UPLOAD_FOLDER, img_name])
        print ("Save it to:", img_path)
        upload.save(img_path)

    result, segment_time = fashion_segmenter.segment(img_path)
    masks = result['masks']
    class_ids = result['class_ids']
    boxes = utils.extract_bboxes(masks)
    class_names = ["_BG_", "short sleeve top", "long sleeve top", "short sleeve outwear", "long sleeve outwear", "vest", "sling", "shorts", "trousers", "skirt", "short sleeve dress", "long sleeve dress", "vest dress", "sling dress"]
    image = skimage.io.imread(img_path)
    draw_predict(image, img_name, boxes, masks, class_ids, class_names)
    label_scores = []
    for class_id, score in zip(class_ids, result['scores']):
        label_scores.append((class_names[class_id], score))
    return flask.render_template('index.html', has_result=True, img_name='predict_'+img_name,\
        label_scores=label_scores, segment_time=segment_time)


@app.route('/upload/<filename>')
def send_image(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


class FashionSegmenter(object):
    def __init__(self, weight_path, config, logs):
        logging.info('Loading net and associated files...')
        self.model = modellib.MaskRCNN(mode="inference", config=config, model_dir=logs)
        self.model.load_weights(weight_path, by_name=True)

    def segment(self, image_filename):
        try:
            image = skimage.io.imread(image_filename)
            starttime = time.time()
            result = self.model.detect([image])[0]
            endtime = time.time()
            rtn = (result, '%.3f' % (endtime - starttime))
            return rtn

        except Exception as err:
            logging.info('Classification error: %s', err)
            return (False, 'Something went wrong when classifying the '
                           'image. Maybe try another one?')

class InferenceConfig(DeepFashion2Config):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    DETECTION_MIN_CONFIDENCE = 0.5


def start_tornado(app, port=5000):
    http_server = tornado.httpserver.HTTPServer(
        tornado.wsgi.WSGIContainer(app))
    http_server.listen(port)
    print("Tornado server starting on port {}".format(port))
    tornado.ioloop.IOLoop.instance().start()


if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(os.path.join(REPO_DIRNAME, UPLOAD_FOLDER))

    config = InferenceConfig()
    weight = 'runs/DETECTION_MIN_CONFIDENCE=0.7/mask_rcnn_fashion_0119.h5'

    global fashion_segmenter
    fashion_segmenter = FashionSegmenter(weight, config, logs='runs')

    start_tornado(app)
