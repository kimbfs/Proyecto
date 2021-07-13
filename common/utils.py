import os
import numpy as np
import time
import cv2, colorsys
from PIL import Image
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb

from common.backbones.efficientnet import swish
from common.backbones.mobilenet_v3 import hard_sigmoid, hard_swish


def optimize_tf_gpu(tf, K):
    if tf.__version__.startswith('2'):
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # Currently, memory growth needs to be the same across GPUs
                for gpu in gpus:
                    #tf.config.experimental.set_virtual_device_configuration(gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=10000)])
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(e)
    else:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True   #dynamic alloc GPU resource
        config.gpu_options.per_process_gpu_memory_fraction = 0.9  #GPU memory threshold 0.3
        session = tf.Session(config=config)

        # set session
        K.set_session(session)


def get_custom_objects():
    '''
    form up a custom_objects dict so that the customized
    layer/function call could be correctly parsed when keras
    .h5 model is loading or converting
    '''
    custom_objects_dict = {
        'tf': tf,
        'swish': swish,
        'hard_sigmoid': hard_sigmoid,
        'hard_swish': hard_swish,
        'mish': mish
    }

    return custom_objects_dict


def get_multiscale_list():
    input_shape_list = [(320,320), (352,352), (384,384), (416,416), (448,448), (480,480), (512,512), (544,544), (576,576), (608,608)]

    return input_shape_list


def resize_anchors(base_anchors, target_shape, base_shape=(416,416)):
    '''
    original anchor size is clustered from COCO dataset
    under input shape (416,416). We need to resize it to
    our train input shape for better performance
    '''
    return np.around(base_anchors*target_shape[::-1]/base_shape[::-1])


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    with open(anchors_path) as f:
        anchors = f.readline()
    anchors = [float(x) for x in anchors.split(',')]
    return np.array(anchors).reshape(-1, 2)

def get_dataset(annotation_file, shuffle=True):
    with open(annotation_file) as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]

    if shuffle:
        np.random.seed(int(time.time()))
        np.random.shuffle(lines)

    return lines

def draw_label(image, text, color, coords):
    font = cv2.FONT_HERSHEY_PLAIN
    font_scale = 1.
    (text_width, text_height) = cv2.getTextSize(text, font, fontScale=font_scale, thickness=1)[0]

    padding = 5
    rect_height = text_height + padding * 2
    rect_width = text_width + padding * 2

    (x, y) = coords

    cv2.rectangle(image, (x, y), (x + rect_width, y - rect_height), color, cv2.FILLED)
    cv2.putText(image, text, (x + padding, y - text_height + padding), font,
                fontScale=font_scale,
                color=(255, 255, 255),
                lineType=cv2.LINE_AA)

    return image

def draw_boxes(image, boxes, classes, scores, class_names, show_score=True):
    if boxes is None or len(boxes) == 0:
        return image
    if classes is None or len(classes) == 0:
        return image

    salida=[]
    for box, cls, score in zip(boxes, classes, scores):
        xmin, ymin, xmax, ymax = map(int, box)
        
        #new section for color detection 
        imageFrame = image[(ymin+10):(ymax-10),(xmin+10):(xmax-10)]

        # HSV(hue-saturation-value)
        
        hsvFrame = imageFrame
        hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)#

        # define colors
        red_lower = np.array([110, 100, 100], np.uint8)
        red_upper = np.array([130, 255, 255], np.uint8)
        red_mask = cv2.inRange(hsvFrame, red_lower, red_upper)
        
        green_lower = np.array([50, 100, 100], np.uint8)
        green_upper = np.array([70, 255, 210], np.uint8)
        green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)
        
        blue_lower = np.array([0, 100, 100], np.uint8)
        blue_upper = np.array([10, 255, 255], np.uint8)
        blue_mask = cv2.inRange(hsvFrame, blue_lower, blue_upper)
        
        yellow_lower = np.array([80, 100, 100], np.uint8)
        yellow_upper = np.array([100, 255, 255], np.uint8)
        yellow_mask = cv2.inRange(hsvFrame, yellow_lower, yellow_upper)    
        
        cyan_lower = np.array([20, 100, 100], np.uint8)
        cyan_upper = np.array([40, 255, 255], np.uint8)
        cyan_mask = cv2.inRange(hsvFrame, cyan_lower, cyan_upper)    
        
        magenta_lower = np.array([140, 100, 100], np.uint8)
        magenta_upper = np.array([160, 255, 255], np.uint8)
        magenta_mask = cv2.inRange(hsvFrame, magenta_lower, magenta_upper)
        
        kernal = np.ones((8, 8), "uint8")
        
        red_mask = cv2.dilate(red_mask, kernal)
        res_red = cv2.bitwise_and(imageFrame, imageFrame, 
                                  mask = red_mask)
        
        green_mask = cv2.dilate(green_mask, kernal)
        res_green = cv2.bitwise_and(imageFrame, imageFrame,
                                    mask = green_mask)
        
        blue_mask = cv2.dilate(blue_mask, kernal)
        res_blue = cv2.bitwise_and(imageFrame, imageFrame,
                                   mask = blue_mask)
                                   
        yellow_mask = cv2.dilate(yellow_mask, kernal)
        res_yellow = cv2.bitwise_and(imageFrame, imageFrame,
                                   mask = yellow_mask)
                                   
        cyan_mask = cv2.dilate(cyan_mask, kernal)
        res_cyan = cv2.bitwise_and(imageFrame, imageFrame,
                                   mask = cyan_mask)
                                   
        magenta_mask = cv2.dilate(magenta_mask, kernal)
        res_magenta = cv2.bitwise_and(imageFrame, imageFrame,
                                   mask = magenta_mask)
        
        r=b=g=c=ye=m="" 
        area1=area2=area3=area4=area5=area6=0
        
        contours, hierarchy = cv2.findContours(red_mask,
                                               cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_SIMPLE)
               
        for pic, contour in enumerate(contours):
            area1 = cv2.contourArea(contour)
            if(area1 > 1600):
                x, y, w, h = cv2.boundingRect(contour)
                imageFrame = cv2.rectangle(imageFrame, (x, y),
                                           (x + w, y + h), 
                                           (0, 0, 0), 2)
                cv2.putText(imageFrame, "Red", (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            (0, 0, 0))
                r="red"
        
        contours, hierarchy = cv2.findContours(green_mask,
                                               cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_SIMPLE)
        for pic, contour in enumerate(contours):
            area2 = cv2.contourArea(contour)
            if(area2 > 1600):
                x, y, w, h = cv2.boundingRect(contour)
                imageFrame = cv2.rectangle(imageFrame, (x, y), 
                                           (x + w, y + h),
                                           (0, 0, 0), 2)
                cv2.putText(imageFrame, "Green", (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            1.0, (0, 0, 0))
                g="green"
        
        contours, hierarchy = cv2.findContours(blue_mask,
                                               cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_SIMPLE)
        for pic, contour in enumerate(contours):
            area3 = cv2.contourArea(contour)
            if(area3 > 1600):
                x, y, w, h = cv2.boundingRect(contour)
                imageFrame = cv2.rectangle(imageFrame, (x, y),
                                           (x + w, y + h),
                                           (0, 0, 0), 2)
                cv2.putText(imageFrame, "Blue", (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (0, 0, 0))
                b="blue"
                            
        contours, hierarchy = cv2.findContours(yellow_mask,
                                               cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_SIMPLE)
        for pic, contour in enumerate(contours):
            area4 = cv2.contourArea(contour)
            if(area4 > 1600):
                x, y, w, h = cv2.boundingRect(contour)
                imageFrame = cv2.rectangle(imageFrame, (x, y),
                                           (x + w, y + h),
                                           (0, 0, 0), 2)
                cv2.putText(imageFrame, "Yellow", (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (0, 0, 0))
                ye="yellow"
                            
        contours, hierarchy = cv2.findContours(cyan_mask,
                                               cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_SIMPLE)
        for pic, contour in enumerate(contours):
            area5 = cv2.contourArea(contour)
            if(area5 > 1600):
                x, y, w, h = cv2.boundingRect(contour)
                imageFrame = cv2.rectangle(imageFrame, (x, y),
                                           (x + w, y + h),
                                           (0, 0, 0), 2)
                cv2.putText(imageFrame, "Cyan", (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1.0, (0, 0, 0))
                c="cyan"
                            
        contours, hierarchy = cv2.findContours(magenta_mask,
                                               cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_SIMPLE)
        for pic, contour in enumerate(contours):
            area6 = cv2.contourArea(contour)
            if(area6 > 1600):
                x, y, w, h = cv2.boundingRect(contour)
                imageFrame = cv2.rectangle(imageFrame, (x, y), 
                                           (x + w, y + h),
                                           (0, 0, 0), 2)
                cv2.putText(imageFrame, "Magenta", (x, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            1.0, (0, 0, 0))
                m="magenta"
        
        class_name = class_names[cls]
        name=[class_name]
        
        if ((area3 < area1) & (area2 < area1) & (area4 < area1) & (area5 < area1) & (area6 < area1)):
            colo=[r]
            
        elif ((area3 < area2) & (area1 < area2) & (area4 < area2) & (area5 < area2) & (area6 < area2)):
            colo=[g]
            
        elif ((area1 < area3) & (area2 < area3) & (area4 < area3) & (area5 < area3) & (area6 < area3)):
            colo=[b]
            
        elif ((area3 < area4) & (area2 < area4) & (area1 < area4) & (area5 < area4) & (area6 < area4)):
            colo=[ye]
            
        elif ((area3 < area5) & (area2 < area5) & (area1 < area5) & (area4 < area5) & (area6 < area5)):
            colo=[c]
        
        elif ((area3 < area6) & (area2 < area6) & (area1 < area6) & (area5 < area6) & (area4 < area6)):
            colo=[m]
            
        else:
            colo=[""]
        
        sal= np.concatenate((colo,name))
        salida=np.concatenate((salida,sal))
        
        if show_score:
            label = '{} {:.2f}'.format(class_name, score)
        else:
            label = '{}'.format(class_name)
            
        color = (0,0,0)
        
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 1, cv2.LINE_AA)
        image = draw_label(image, label, color, (xmin, ymin))

    return image,salida

