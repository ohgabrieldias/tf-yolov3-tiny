import numpy as np
import tensorflow as tf

IMAGE_SIZE = 416
GRID_SIZES = [13, 26]
grid_size = 13
stride = IMAGE_SIZE / grid_size
ANCHORS_FLIPPING = [
    [(125, 141), (89, 225), (227, 237)],  # Anchors para escala 13x13 (objetos grandes)
    [(30, 78), (61, 88), (55, 145)]      # Anchors para escala 26x26 (objetos médios/pequenos)
]


anchors_tf = tf.constant(ANCHORS_FLIPPING, dtype=tf.float32) #
# normaliza os anchors entre 0 e 1
anchors_tf = anchors_tf / tf.constant(IMAGE_SIZE, dtype=tf.float32)
#print(f"anchors_tf: {anchors_tf}")
gs_tf = tf.constant(GRID_SIZES, dtype=tf.float32)


s_reshaped = tf.reshape(gs_tf, [-1, 1, 1])

# Repetir para as dimensões [3, 3, 2]
s_repeated = tf.tile(s_reshaped, [1, 3, 2])

# Escalar as âncoras
scaled_anchors_tf = anchors_tf * s_repeated
def iou_wh(pred_wh, true_wh):
    """
    Calcula a IoU entre duas caixas delimitadoras (width, height) assumindo que ambas estão
    centradas na mesma coordenada.

    Parâmetros:
    - pred_wh: Tensor shape (..., 2), largura e altura da predição
    - true_wh: Tensor shape (..., 2), largura e altura da ground truth

    Retorna:
    - iou: Tensor com o valor da IoU
    """
    pred_wh = tf.convert_to_tensor(pred_wh, dtype=tf.float32)
    true_wh = tf.convert_to_tensor(true_wh, dtype=tf.float32)

    # Interseção: largura e altura mínimas entre as duas caixas
    inter_wh = tf.minimum(pred_wh, true_wh)
    inter_area = inter_wh[..., 0] * inter_wh[..., 1]

    # Áreas individuais
    pred_area = pred_wh[..., 0] * pred_wh[..., 1]
    true_area = true_wh[..., 0] * true_wh[..., 1]

    # União
    union_area = pred_area + true_area - inter_area

    iou = inter_area / (union_area + 1e-6)  # evita divisão por zero
    return iou

bboxes = np.array([
    [0, 0.40865384615384615, 0.41225961538461536, 0.125, 0.5036057692307693],
    [0, 0.10817307692307693, 0.390625, 0.1502403846153846, 0.24158653846153846],
    [0, 0.38461538461538464, 0.4795673076923077, 0.14182692307692307, 0.24158653846153846]
])


bbox = tf.constant(bboxes[0], dtype=tf.float32)  # Simulando uma caixa delimitadora
_, x, y, w, h = bbox

cx = x * IMAGE_SIZE
cy = y * IMAGE_SIZE

grid_x = int(cx // stride)  # índice da célula na largura
grid_y = int(cy // stride) 

tx = (cx / stride) - grid_x
ty = (cy / stride) - grid_y

tw = w * grid_size
th = h * grid_size

true = np.array([1, tx, ty, tw, th])

print(f"true: {true}")
pred = bbox * 0.5

pred_tf = tf.constant(pred, dtype=tf.float32) 
print(f"pred_tf: {pred_tf}")
anchors = scaled_anchors_tf[0]

box_preds_wh = tf.exp(pred_tf[3:5]) * anchors  

mse = tf.keras.losses.MeanSquaredError(reduction='none')
#print(iou_wh(box_preds_wh, true[3:5]))
print(f"box_preds_wh: {box_preds_wh}")


tw = tf.math.log(true[3:5] / anchors)
print(tf.exp(tw)* anchors)
print(f"tw: {tw}")