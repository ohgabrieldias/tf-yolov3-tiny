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
DATASET_PATH = "flipping_bird_dataset"
CLASSES_FILE = f"{DATASET_PATH}/train/_darknet.labels"
NUM_CLASSES = len(open(CLASSES_FILE).readlines())

# ============================ #
#  PARÂMETROS DE TREINAMENTO  #
# ============================ #
EPOCHS = 250
QAT_EPOCHS = 20
SAVE_FREQ = 10
BATCH_SIZE = 10                     # Ajuste conforme sua GPU (MX330)
LEARNING_RATE = 1e-4                # Taxa de aprendizado inicial
DECAY_STEPS = 500                  # Decaimento a cada 1000 passos
DECAY_RATE = 0.95                   # Fator de redução da taxa
IOU_THRESHOLD = 0.5

LAMBDA_CLASS = 5.0
LAMBDA_NO_OBJECT = 0.5

LOAD_MODEL = False
MODEL_NAME = "saved_models/yolov3_epoch_210_trainloss_16.1000_valloss_17.7238.keras"

# ============================ #
#  ANCHORS E ESCALAS          #
# ============================ #
STRIDES = [8, 16]  # Escalas do YOLO
GRID_SIZES = [13, 26]

# Anchors originais (em pixels)
ANCHORS = [
    [(125, 141), (89, 225), (227, 237)],  # Anchors para escala 13x13 (objetos grandes)
    [(30, 78), (61, 88), (55, 145)]       # Anchors para escala 26x26 (objetos médios/pequenos)
]

anchors_tf = tf.constant(ANCHORS, dtype=tf.float32) #
# normaliza os anchors entre 0 e 1
anchors_tf = anchors_tf / tf.constant(IMAGE_SIZE, dtype=tf.float32)