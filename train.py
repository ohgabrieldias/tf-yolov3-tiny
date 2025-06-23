import os
import time
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from config import *
from Dataset import YOLODatasetTF
from Loss import yolo_tiny_loss
from yolov3_tiny import yolov3_tiny
from tensorflow.keras.optimizers import Adam, schedules
from tqdm import tqdm

def configure_gpu():
    """Configura a GPU para melhor desempenho."""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Ativa crescimento dinâmico de memória e mixed-precision
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print("✅ GPU configurada com mixed-precision (FP16).")
        except RuntimeError as e:
            print(f"❌ Erro na GPU: {e}")

def apply_qat_to_model(model):
    """Aplica Quantization Aware Training ao modelo funcional."""
    annotated_model = tfmot.quantization.keras.quantize_annotate_model(model)
    quant_aware_model = tfmot.quantization.keras.quantize_apply(annotated_model)
    print("✅ QAT aplicado com sucesso ao modelo funcional.")
    return quant_aware_model


def create_optimizer():
    """Cria otimizador com decaimento exponencial."""
    lr_schedule = schedules.ExponentialDecay(
        initial_learning_rate=LEARNING_RATE,
        decay_steps=DECAY_STEPS,
        decay_rate=DECAY_RATE,
        staircase=True
    )
    return Adam(learning_rate=lr_schedule)

def load_or_create_model():
    """Carrega um modelo existente ou cria um novo."""
    if LOAD_MODEL:
        print(f"🔍 Carregando modelo: {MODEL_NAME}")
        model = tf.keras.models.load_model(
            MODEL_NAME,
            custom_objects={'yolo_loss': yolo_tiny_loss},
            compile=False
        )
        initial_epoch = int(MODEL_NAME.split('epoch_')[1].split('_')[0])
        print(f"↩️ Retomando treino da época {initial_epoch}")
    else:
        print("🆕 Criando novo modelo YOLOv3-Tiny")
        model = yolov3_tiny()
        initial_epoch = 0
    return model, initial_epoch

@tf.function  # Compila para execução mais rápida
def train_step_compiled(model, optimizer, images, targets):
    return train_step(model, optimizer, images, targets)

def train_step(model, optimizer, images, targets):
    """Executa um passo de treinamento para ambas as escalas."""
    with tf.GradientTape() as tape:
        pred_small, pred_medium = model(images, training=True)
        target_small, target_medium = targets

        loss_small = yolo_tiny_loss(pred_small, target_small, S=GRID_SIZES[0])
        loss_medium = yolo_tiny_loss(pred_medium, target_medium, S=GRID_SIZES[1])

        loss = loss_small + loss_medium

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


def evaluate_model(model, dataset):
    """Avalia o modelo no conjunto de validação."""
    val_loss = 0.0
    val_steps = 0
    for images_val, targets_val in dataset:
        pred_small, pred_medium = model(images_val, training=False)
        target_small, target_medium = targets_val

        loss_small = yolo_tiny_loss(pred_small, target_small, S=GRID_SIZES[0])
        loss_medium = yolo_tiny_loss(pred_medium, target_medium, S=GRID_SIZES[1])

        val_loss += (loss_small + loss_medium)
        val_steps += 1
    return val_loss / max(val_steps, 1)


def save_model(model, epoch, train_loss, val_loss):
    """Salva o modelo com métricas."""
    os.makedirs('saved_models', exist_ok=True)
    model_name = (
        f"saved_models/yolov3_epoch_{epoch}_"
        f"trainloss_{train_loss:.4f}_valloss_{val_loss:.4f}.keras"
    )
    model.save(model_name, save_format='tf')
    return model_name

def export_tflite(model):
    """Exporta o modelo para TFLite quantizado."""
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    quantized_tflite_model = converter.convert()
    
    tflite_path = 'quantized_model.tflite'
    with open(tflite_path, 'wb') as f:
        f.write(quantized_tflite_model)
    return tflite_path

def main():
    # Configuração inicial
    configure_gpu()
    model, initial_epoch = load_or_create_model()
    model = apply_qat_to_model(model)
    optimizer = create_optimizer()

    # Preparação do dataset
    train_dataset = (
        YOLODatasetTF(DATASET_PATH, subset='train')
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
        .cache()
    )
    val_dataset = (
        YOLODatasetTF(DATASET_PATH, subset='valid')
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    # Loop de treinamento
    for epoch in range(initial_epoch, EPOCHS):
        print(f"\n🎯 Época {epoch + 1}/{EPOCHS}")
        start_time = time.time()
        epoch_loss = 0.0
        steps = 0

        # Fase de treino
        progress_bar = tqdm(train_dataset, desc="Treinando", total=len(train_dataset))
        for images, targets in progress_bar:
            loss = train_step_compiled(model, optimizer, images, targets)

            epoch_loss += loss.numpy()
            steps += 1
            progress_bar.set_postfix({"Loss": f"{loss.numpy():.4f}"})

        avg_train_loss = epoch_loss / steps
        epoch_time = time.time() - start_time

        # Fase de validação
        avg_val_loss = evaluate_model(model, val_dataset)
        
        print(f"✅ Treino | Loss: {avg_train_loss:.4f} | "
              f"Validação | Loss: {avg_val_loss:.4f} | "
              f"Tempo: {epoch_time:.2f}s")

        # Salva o modelo periodicamente
        if (epoch + 1) % SAVE_FREQ == 0 or (epoch + 1) == EPOCHS:
            model_path = save_model(model, epoch+1, avg_train_loss, avg_val_loss)
            print(f"💾 Modelo salvo em: {model_path}")

    # Exportação final
    tflite_path = export_tflite(model)
    print(f"\n🚀 Modelo TFLite quantizado exportado: {tflite_path}")

if __name__ == "__main__":
    main()