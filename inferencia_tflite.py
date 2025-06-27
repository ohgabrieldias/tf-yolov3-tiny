import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from config import *

# ========== Funções ==========

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (IMAGE_SIZE, IMAGE_SIZE))
    img_input = img_resized / 255.0
    img_input = np.expand_dims(img_input.astype(np.float32), axis=0)
    return img_input, img_resized

def decode_output(pred, anchors, scale):
    batch_size, grid_size, _, _, _ = pred.shape
    boxes = []
    pred = pred[0]  # remove batch dim → shape (3, S, S, 6)

    for anchor_idx in range(3):
        for i in range(scale):
            for j in range(scale):
                tx, ty, tw, th, obj, cls = pred[anchor_idx, i, j]

                objectness = sigmoid(obj)
                if objectness < CONF_THRESHOLD:
                    continue
                stride = IMAGE_SIZE / scale
                cx = (sigmoid(tx) + j) * stride
                cy = (sigmoid(ty) + i) * stride
                w = np.exp(tw) * anchors[anchor_idx][0] * stride
                h = np.exp(th) * anchors[anchor_idx][1] * stride

                x1 = (cx - w / 2)
                y1 = (cy - h / 2)
                x2 = (cx + w / 2)
                y2 = (cy + h / 2)

                boxes.append([x1, y1, x2, y2, objectness, 0])  # 0: class id
    return boxes

def nms(boxes, iou_thresh=0.4):
    if not boxes:
        return []
    boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
    final_boxes = []

    while boxes:
        chosen = boxes.pop(0)
        final_boxes.append(chosen)
        boxes = [box for box in boxes if iou(chosen, box) < iou_thresh]
    return final_boxes

def iou(box1, box2):
    xa1, ya1, xa2, ya2 = box1[:4]
    xb1, yb1, xb2, yb2 = box2[:4]

    inter_x1 = max(xa1, xb1)
    inter_y1 = max(ya1, yb1)
    inter_x2 = min(xa2, xb2)
    inter_y2 = min(ya2, yb2)

    inter_area = max(inter_x2 - inter_x1, 0) * max(inter_y2 - inter_y1, 0)
    area_a = (xa2 - xa1) * (ya2 - ya1)
    area_b = (xb2 - xb1) * (yb2 - yb1)
    union_area = area_a + area_b - inter_area

    return inter_area / union_area if union_area > 0 else 0

def draw_boxes(image, boxes):
    for x1, y1, x2, y2, score, cls in boxes:
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(image, f"{score:.2f}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    return image

def reshape_output(output, S):
    """
    Transforma (1, S, S, 18) em (1, 3, S, S, 6)
    """
    return output.reshape(1, S, S, 3, 6).transpose(0, 3, 1, 2, 4)

# ========== Carregar modelo TFLite ==========
interpreter = tf.lite.Interpreter(model_path="quantized_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ========== Inference ==========
img_name = "000001"
img_path = f"{DATASET_PATH}/train/{img_name}.jpg"

image_input, image_draw = preprocess_image(img_path)

interpreter.set_tensor(input_details[0]['index'], image_input)
interpreter.invoke()

output_13 = interpreter.get_tensor(output_details[0]['index'])  # (1, 13, 13, 18)
output_26 = interpreter.get_tensor(output_details[1]['index'])  # (1, 13, 13, 18)

output_13 = reshape_output(output_13, 13)
output_26 = reshape_output(output_26, 26)

boxes_13 = decode_output(output_13, scaled_anchors_tf[0], 13)
boxes_26 = decode_output(output_26, scaled_anchors_tf[1], 26)

all_boxes = boxes_26
final_boxes = nms(all_boxes, NMS_THRESHOLD)

# ========== Visualização ==========
image_with_boxes = draw_boxes(image_draw, final_boxes)
plt.imshow(cv2.cvtColor(image_with_boxes, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title("Resultado")
plt.show()
