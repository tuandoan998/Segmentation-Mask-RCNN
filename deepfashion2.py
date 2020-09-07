"""
Mask R-CNN
Train on the DeepFashion2 dataset and implement color splash effect.

Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla

------------------------------------------------------------

Usage: import the module (see Jupyter notebooks for examples), or run from
       the command line as such:

    # Train a new model starting from pre-trained COCO weights
    python3 deepfashion2.py train --dataset=/path/to/deepfashion2/dataset --weights=coco

    # Resume training a model that you had trained earlier
    python3 deepfashion2.py train --dataset=/path/to/deepfashion2/dataset --weights=last

    # Train a new model starting from ImageNet weights
    python3 deepfashion2.py train --dataset=/path/to/deepfashion2/dataset --weights=imagenet

    # Apply color splash to an image
    python3 deepfashion2.py splash --weights=/path/to/weights/file.h5 --image=<URL or path to file>

    # Apply color splash to video using the last weights you trained
    python3 deepfashion2.py splash --weights=last --video=<URL or path to file>
"""

import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import glob
from tqdm.notebook import tqdm
import imgaug
import time

# Import Mask RCNN
Mask_RCNN_DIR = os.path.abspath("Mask_RCNN")
sys.path.append(Mask_RCNN_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
from mrcnn import model as modellib

# Path to trained weights file
COCO_WEIGHTS_PATH = "mask_rcnn_coco.h5"

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
DEFAULT_LOGS_DIR = "runs"

############################################################
#  Configurations
############################################################


class DeepFashion2Config(Config):
    """Configuration for training on the DeepFashion2 dataset.
    Derives from the base Config class and overrides some values.
    """
    # Give the configuration a recognizable name
    NAME = "fashion"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 13  # background + 13 DeepFashion2

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100

    # Skip detections with < 70% confidence
    DETECTION_MIN_CONFIDENCE = 0.7


############################################################
#  Dataset
############################################################

class DeepFashion2Dataset(utils.Dataset):

    def load_fashion(self, dataset_dir, subset, limit_categories=True):
        """Load a subset of the DeepFashion2 dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        # Add classes. We have only one class to add.
        self.add_class("fashion", 1, "short sleeve top")
        self.add_class("fashion", 2, "long sleeve top")
        self.add_class("fashion", 3, "short sleeve outwear")
        self.add_class("fashion", 4, "long sleeve outwear")
        self.add_class("fashion", 5, "vest")
        self.add_class("fashion", 6, "sling")
        self.add_class("fashion", 7, "shorts")
        self.add_class("fashion", 8, "trousers")
        self.add_class("fashion", 9, "skirt")
        self.add_class("fashion", 10, "short sleeve dress")
        self.add_class("fashion", 11, "long sleeve dress")
        self.add_class("fashion", 12, "vest dress")
        self.add_class("fashion", 13, "sling dress")

        # Train or validation dataset?
        assert subset in ["train", "validation"]

        img_paths = glob.glob(os.path.join(dataset_dir, subset, 'image', '*.jpg'))
        anno_paths = [img_path.replace('image', 'annos').replace('jpg', 'json') for img_path in img_paths]
        assert (len(img_paths) == len(anno_paths))
        
        # Limit number image of each class: 100
        if limit_categories:
            if subset=='train':
                category_limit = 200
            elif subset=='validation':
                category_limit = 50
        num_categories = np.zeros((13,), dtype=int)

        # Add images
        print(f'\nLoading images and annotations of {subset} ... ([Number masks of each category])')
        for i in tqdm(range(len(img_paths)), position=0, leave=True):
            if i%2000==0:
                print(num_categories)
            if limit_categories and ((num_categories>=category_limit).sum() == len(num_categories)):
                break
            
            image = skimage.io.imread(img_paths[i])
            height, width = image.shape[:2]
            with open(anno_paths[i], 'r') as file:
                data = json.load(file)
            polygons = []
            category_ids = []
            for key in data:
                if key == 'source' or key=='pair_id':
                    continue
                else:
                    category_id = data[key]['category_id']
                    category_ids.append(category_id)
                    x = [poly[0::2] for poly in data[key]['segmentation']] # [[x1, x2, ...], [x1, x2, ...], ...]
                    y = [poly[1::2] for poly in data[key]['segmentation']] # [[y1, y2, ...], [y1, y2, ...], ...]
                    polygons.append({'all_points_x': x, 'all_points_y': y})

            if limit_categories and (all(num_categories[id-1]>=category_limit for id in category_ids)):
                continue
            
            for id in category_ids:
                num_categories[id-1] = num_categories[id-1] + 1

            self.add_image(
                "fashion",
                image_id=img_paths[i].split('/')[-1],
                path=img_paths[i],
                width=width, height=height,
                polygons=polygons,
                category_ids=category_ids)
        print(num_categories)

    def load_mask(self, image_id):
        """Generate instance masks for an image.
       Returns:
        masks: A bool array of shape [height, width, instance count] with
            one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a deepfashion2 dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != "fashion":
            return super(self.__class__, self).load_mask(image_id)

        # Convert polygons to a bitmap mask of shape
        # [height, width, instance_count]
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])], dtype=np.uint8)
        for i, polygon in enumerate(info["polygons"]):
            # Get indexes of pixels inside the polygon and set them to 1
            for p_idx in range(len(polygon['all_points_y'])):
                rr, cc = skimage.draw.polygon(polygon['all_points_y'][p_idx], polygon['all_points_x'][p_idx])
                rr[rr > mask.shape[0]-1] = mask.shape[0]-1
                cc[cc > mask.shape[1]-1] = mask.shape[1]-1
                mask[rr, cc, i] = 1
        class_ids = info["category_ids"]

        # Return mask, and array of class IDs of each instance. Since we have
        # one class ID only, we return an array of 1s
        class_ids = np.array(class_ids, dtype=np.int32)
        return mask.astype(np.bool), class_ids

    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == "fashion":
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)


def train(model):
    """Train the model."""
    # Training dataset.
    dataset_train = DeepFashion2Dataset()
    dataset_train.load_fashion(args.dataset, "train")
    dataset_train.prepare()

    # Validation dataset
    dataset_val = DeepFashion2Dataset()
    dataset_val.load_fashion(args.dataset, "validation")
    dataset_val.prepare()

    # Image Augmentation
    # Right/Left flip 50% of the time
    augmentation = imgaug.augmenters.Fliplr(0.5)

    # Training - Stage 1
    print("Training network heads")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=50,
                layers='heads',
                augmentation=augmentation)

    # Training - Stage 2
    # Finetune layers from ResNet stage 4 and up
    print("Fine tune Resnet stage 4 and up")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=100,
                layers='4+',
                augmentation=augmentation)

    # Training - Stage 3
    # Fine tune all layers
    print("Fine tune all layers")
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 50,
                epochs=150,
                layers='all',
                augmentation=augmentation)


def color_splash(image, mask):
    """Apply color splash effect.
    image: RGB image [height, width, 3]
    mask: instance segmentation mask [height, width, instance count]

    Returns result image.
    """
    # Make a grayscale copy of the image. The grayscale copy still
    # has 3 RGB channels, though.
    gray = skimage.color.gray2rgb(skimage.color.rgb2gray(image)) * 255
    # Copy color pixels from the original color image where mask is set
    if mask.shape[-1] > 0:
        # We're treating all instances as one, so collapse the mask into one layer
        mask = (np.sum(mask, -1, keepdims=True) >= 1)
        splash = np.where(mask, image, gray).astype(np.uint8)
    else:
        splash = gray.astype(np.uint8)
    return splash


def detect_and_color_splash(model, image_path=None, video_path=None):
    assert image_path or video_path

    # Image or video?
    if image_path:
        # Run model detection and generate the color splash effect
        print("Running on {}".format(args.image))
        # Read image
        image = skimage.io.imread(args.image)
        # Detect objects
        r = model.detect([image], verbose=1)[0]
        # Color splash
        splash = color_splash(image, r['masks'])
        # Save output
        file_name = "splash_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now())
        skimage.io.imsave(file_name, splash)
        io.imshow(splash)
        plt.show()
    elif video_path:
        import cv2
        # Video capture
        vcapture = cv2.VideoCapture(video_path)
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # Define codec and create video writer
        file_name = "splash_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now())
        vwriter = cv2.VideoWriter(file_name,
                                  cv2.VideoWriter_fourcc(*'MJPG'),
                                  fps, (width, height))

        count = 0
        success = True
        while success:
            print("frame: ", count)
            # Read next image
            success, image = vcapture.read()
            if success:
                # OpenCV returns images as BGR, convert to RGB
                image = image[..., ::-1]
                # Detect objects
                r = model.detect([image], verbose=0)[0]
                # Color splash
                splash = color_splash(image, r['masks'])
                # RGB -> BGR to save image to video
                splash = splash[..., ::-1]
                # Add image to video writer
                vwriter.write(splash)
                count += 1
        vwriter.release()
    print("Saved to ", file_name)


def evaluate(model, dataset, config, limit=0):
    image_ids = dataset.image_ids
    if limit:
        image_ids = image_ids[:limit]
    t_prediction = 0
    t_start = time.time()
    APs = []
    for image_id in image_ids:
        image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset, \
            config, image_id, use_mini_mask=False)
        # image = dataset.load_image(image_id)
        # gt_mask, gt_class_id = dataset.load_mask(image_id)
        # gt_bbox = utils.extract_bboxes(gt_mask)
        t = time.time()
        r = model.detect([image])[0]
        t_prediction += (time.time() - t)
        AP, precisions, recalls, overlaps = utils.compute_ap(gt_bbox, gt_class_id, gt_mask, \
            r['rois'], r['class_ids'], r['scores'], r['masks'])
        APs.append(AP)
    print("Prediction time: {}. Average {}/image".format(t_prediction, t_prediction / len(image_ids)))
    print("Total time: ", time.time() - t_start)
    print("mean Average Precision @ IoU=50: ", np.mean(APs))


############################################################
#  Training
############################################################

if __name__ == '__main__':
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train Mask R-CNN to segment fashion.')
    parser.add_argument("command",
                        metavar="<command>",
                        help="'train' or 'evaluate' or 'splash'")
    parser.add_argument('--dataset', required=False,
                        metavar="/path/to/deepfashion2/dataset/",
                        help='Directory of the DeepFashion2 dataset')
    parser.add_argument('--weights', required=True,
                        metavar="/path/to/weights.h5",
                        help="Path to weights .h5 file or 'coco'")
    parser.add_argument('--logs', required=False,
                        default=DEFAULT_LOGS_DIR,
                        metavar="/path/to/logs/",
                        help='Logs and checkpoints directory (default=runs/)')
    parser.add_argument('--image', required=False,
                        metavar="path or URL to image",
                        help='Image to apply the color splash effect on')
    parser.add_argument('--video', required=False,
                        metavar="path or URL to video",
                        help='Video to apply the color splash effect on')
    parser.add_argument('--limit', required=False,
                        default=500,
                        metavar="<image count>",
                        help='Images to use for evaluation (default=500)')
    args = parser.parse_args()

    # Validate arguments
    if args.command == "train":
        assert args.dataset, "Argument --dataset is required for training"
    elif args.command == "splash":
        assert args.image or args.video, "Provide --image or --video to apply color splash"
    elif args.command == "evaluate":
        assert args.dataset, "Argument --dataset is required for evaluation"

    print("Weights: ", args.weights)
    print("Dataset: ", args.dataset)
    print("Logs: ", args.logs)

    # Configurations
    if args.command == "train":
        config = DeepFashion2Config()
    else:
        class InferenceConfig(DeepFashion2Config):
            # Set batch size to 1 since we'll be running inference on
            # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
            GPU_COUNT = 1
            IMAGES_PER_GPU = 1
            DETECTION_MIN_CONFIDENCE = 0
        config = InferenceConfig()
    config.display()

    # Create model
    if args.command == "train":
        model = modellib.MaskRCNN(mode="training", config=config, model_dir=args.logs)
    else:
        model = modellib.MaskRCNN(mode="inference", config=config, model_dir=args.logs)

    # Select weights file to load
    if args.weights.lower() == "coco":
        weights_path = COCO_WEIGHTS_PATH
        # Download weights file
        if not os.path.exists(weights_path):
            utils.download_trained_weights(weights_path)
    elif args.weights.lower() == "last":
        # Find last trained weights
        weights_path = model.find_last()
    elif args.weights.lower() == "imagenet":
        # Start from ImageNet trained weights
        weights_path = model.get_imagenet_weights()
    else:
        weights_path = args.weights

    # Load weights
    print("Loading weights ", weights_path)
    if args.weights.lower() == "coco":
        # Exclude the last layers because they require a matching number of classes
        model.load_weights(weights_path, by_name=True, exclude=[
            "mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
    else:
        model.load_weights(weights_path, by_name=True)

    # Train or evaluate
    if args.command == "train":
        train(model)
    elif args.command == "evaluate":
        # Validation dataset
        dataset_val = DeepFashion2Dataset()
        dataset_val.load_fashion(args.dataset, "validation", limit_categories=False)
        dataset_val.prepare()
        print("Running evaluation on {} images.".format(args.limit))
        evaluate(model, dataset_val, config, limit=int(args.limit))
    elif args.command == "splash":
        detect_and_color_splash(model, image_path=args.image, video_path=args.video)
    else:
        print("'{}' is not recognized. \nUse 'train' or 'splash'".format(args.command))
