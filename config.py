# ============================ #
#  SUPRESSÃO DE LOGS DO TF    #
# ============================ #
import os
import tensorflow as tf
import numpy as np

# Suprimir mensagens do TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'             # 0=all, 1=info, 2=warnings, 3=errors
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'            # Desativa mensagens do oneDNN (opcional)
tf.get_logger().setLevel('ERROR')                    # Suprime logs adicionais

# ============================ #
#  CONFIGURAÇÃO DA GPU        #
# ============================ #
gpus = tf.config.experimental.list_physical_devices('GPU')
print(f'GPUs disponíveis: {gpus}')

if len(gpus) > 0:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except RuntimeError:
        pass

# ============================ #
#  PARÂMETROS DO DATASET      #
# ============================ #
IMAGE_SIZE = 416  # Tamanho padrão para YOLO
INPUT_SIZE = (416, 416)             # Tamanho da imagem de entrada
INPUT_SHAPE = (416, 416, 3)

# ============================ #
#  PARÂMETROS DE TREINAMENTO  #
# ============================ #
# Quantization control
USE_QAT = False  # Set to False to disable quantization
EPOCHS = 110
QAT_EPOCHS = 20
SAVE_FREQ = 15
BATCH_SIZE = 32
LEARNING_RATE = 1e-4                # Taxa de aprendizado inicial
DECAY_STEPS = 1000                  # Decaimento a cada 1000 passos
DECAY_RATE = 0.95                   # Fator de redução da taxa
CONF_THRESHOLD = 0.8
NMS_THRESHOLD = 0.05

LAMBDA_COORD = 5.0
LAMBDA_NOOBJ = 0.5

LOAD_MODEL = False
MODEL_NAME = "saved_models/yolov3_epoch_210_trainloss_16.1000_valloss_17.7238.keras"

# ============================ #
#  ANCHORS E ESCALAS          #
# ============================ #
STRIDES = [32,16]  # Escalas do YOLO
GRID_SIZES = [13, 26]  # Tamanhos das grades correspondentes às escalas

# Anchors originais (em pixels)
ANCHORS_FLIPPING = [
    [(125, 141), (89, 225), (227, 237)],  # Anchors para escala 13x13 (objetos grandes)
    [(30, 78), (61, 88), (55, 145)]      # Anchors para escala 26x26 (objetos médios/pequenos)
]

anchors_tf = tf.constant(ANCHORS_FLIPPING, dtype=tf.float32) #
# normaliza os anchors entre 0 e 1
anchors_tf = anchors_tf / tf.constant(IMAGE_SIZE, dtype=tf.float32)
gs_tf = tf.constant(GRID_SIZES, dtype=tf.float32)


s_reshaped = tf.reshape(gs_tf, [-1, 1, 1])

# Repetir para as dimensões [3, 3, 2]
s_repeated = tf.tile(s_reshaped, [1, 3, 2])

# Escalar as âncoras
scaled_anchors_tf = anchors_tf * s_repeated

DATASET = "FLIPPING"
DATASET_PATH = f"{DATASET}"

# Configurações específicas para cada dataset
DATASET_CONFIG = {
    "PASCAL_VOC": {
        "CLASSES": [
                    "aeroplane",
                    "bicycle",
                    "bird",
                    "boat",
                    "bottle",
                    "bus",
                    "car",
                    "cat",
                    "chair",
                    "cow",
                    "diningtable",
                    "dog",
                    "horse",
                    "motorbike",
                    "person",
                    "pottedplant",
                    "sheep",
                    "sofa",
                    "train",
                    "tvmonitor"
                ],
        "IMG_DIR": "images",
        "LABEL_DIR": "labels",
        "USE_CSV": True,
        "TRAIN_CSV": "train.csv",
        "VAL_CSV": "test.csv"
    },
    "FLIPPING": {
        "CLASSES": ["dedo_do_meio"],
        "TRAIN_DIR": "train",
        "VAL_DIR": "valid",
        "USE_CSV": False
    }
}

# Configurações ativas baseadas no DATASET selecionado
CURRENT_CONFIG = DATASET_CONFIG[DATASET]
CLASSES = CURRENT_CONFIG["CLASSES"]
NUM_CLASSES = len(CLASSES)