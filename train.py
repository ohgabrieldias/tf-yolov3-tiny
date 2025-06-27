import os
import time
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from config import *
from Dataset import YOLODatasetTF
from Loss import yolo_loss
from yolov3_tiny import yolov3_tiny
from tensorflow.keras.optimizers import Adam, schedules
from tqdm import tqdm

def configure_gpu():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            policy = tf.keras.mixed_precision.Policy('mixed_float16')
            tf.keras.mixed_precision.set_global_policy(policy)
            print("✅ GPU configurada com mixed-precision (FP16).")
        except RuntimeError as e:
            print(f"❌ Erro na GPU: {e}")

def apply_qat_to_model(model):
    if not USE_QAT:
        print("⏭️ QAT desativado na configuração. Modelo não será quantizado.")
        return model
        
    annotated_model = tfmot.quantization.keras.quantize_annotate_model(model)
    quant_aware_model = tfmot.quantization.keras.quantize_apply(annotated_model)
    print("✅ QAT aplicado com sucesso ao modelo funcional.")
    return quant_aware_model

def create_optimizer():
    lr_schedule = schedules.ExponentialDecay(
        initial_learning_rate=LEARNING_RATE,
        decay_steps=DECAY_STEPS,
        decay_rate=DECAY_RATE,
        staircase=True
    )
    return Adam(learning_rate=lr_schedule)

def load_or_create_model():
    if LOAD_MODEL:
        print(f"🔍 Carregando modelo: {MODEL_NAME}")
        model = tf.keras.models.load_model(
            MODEL_NAME,
            custom_objects={'yolo_loss': yolo_loss},
            compile=False
        )
        initial_epoch = int(MODEL_NAME.split('epoch_')[1].split('_')[0])
        print(f"↩️ Retomando treino da época {initial_epoch}")
    else:
        print("🆕 Criando novo modelo YOLOv3-Tiny")
        model = yolov3_tiny()
        initial_epoch = 0
    return model, initial_epoch

@tf.function
def train_step_compiled(model, optimizer, images, targets):
    return train_step(model, optimizer, images, targets)

def train_step(model, optimizer, images, targets):
    with tf.GradientTape() as tape:
        pred_small, pred_medium = model(images, training=True)
        target_small, target_medium = targets

        loss_small = yolo_loss(pred_small, target_small, S=GRID_SIZES[0], anchors=scaled_anchors_tf[0])
        loss_medium = yolo_loss(pred_medium, target_medium, S=GRID_SIZES[1], anchors=scaled_anchors_tf[1])
        loss_total = loss_small + loss_medium

    gradients = tape.gradient(loss_total, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss_total, loss_small, loss_medium

def evaluate_model(model, dataset):
    val_loss_total = 0.0
    val_loss_small = 0.0
    val_loss_medium = 0.0
    val_steps = 0

    for images_val, targets_val in dataset:
        pred_small, pred_medium = model(images_val, training=False)
        target_small, target_medium = targets_val

        loss_small = yolo_loss(pred_small, target_small, S=GRID_SIZES[0], anchors=scaled_anchors_tf[0])
        loss_medium = yolo_loss(pred_medium, target_medium, S=GRID_SIZES[1], anchors=scaled_anchors_tf[1])
        loss_total = loss_small + loss_medium

        val_loss_total += loss_total
        val_loss_small += loss_small
        val_loss_medium += loss_medium
        val_steps += 1

    return (
        val_loss_total / val_steps,
        val_loss_small / val_steps,
        val_loss_medium / val_steps
    )

def save_model(model, epoch, train_loss, val_loss):
    """Salva o modelo com nome compacto."""
    os.makedirs('saved_models', exist_ok=True)
    model_name = (
        f"saved_models/yv3tiny_e{epoch}_"
        f"tl{train_loss:.1f}_vl{val_loss:.1f}.keras"
    )
    model.save(model_name, save_format='tf')
    return model_name


def export_tflite(model):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    if USE_QAT:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
        print("🔧 Exportando modelo TFLite quantizado...")
    else:
        print("🔧 Exportando modelo TFLite padrão (não quantizado)...")
    
    quantized_tflite_model = converter.convert()

    tflite_path = 'quantized_model.tflite' if USE_QAT else 'model.tflite'
    with open(tflite_path, 'wb') as f:
        f.write(quantized_tflite_model)
    return tflite_path

def main():
    configure_gpu()
    model, initial_epoch = load_or_create_model()
    model = apply_qat_to_model(model)
    optimizer = create_optimizer()

    # Como usar (parte do seu código adaptada)
    train_dataset = (
        YOLODatasetTF(DATASET_PATH, subset='train', config=CURRENT_CONFIG)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
        .shuffle(1000)
    )

    val_dataset = (
        YOLODatasetTF(DATASET_PATH, subset='valid', config=CURRENT_CONFIG)
        .batch(BATCH_SIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    for epoch in range(initial_epoch, EPOCHS):
        print(f"\n🎯 Época {epoch + 1}/{EPOCHS}")
        start_time = time.time()
        epoch_loss_total = 0.0
        epoch_loss_small = 0.0
        epoch_loss_medium = 0.0
        steps = 0

        #progress_bar = tqdm(train_dataset, desc="Treinando", total=len(train_dataset))
        for images, targets in train_dataset:
            loss_total, loss_small, loss_medium = train_step_compiled(model, optimizer, images, targets)

            epoch_loss_total += loss_total.numpy()
            epoch_loss_small += loss_small.numpy()
            epoch_loss_medium += loss_medium.numpy()
            steps += 1

            # progress_bar.set_postfix({
            #     "Total": f"{loss_total.numpy():.1f}",
            #     "13x13": f"{loss_small.numpy():.1f}",
            #     "26x26": f"{loss_medium.numpy():.1f}"
            # })

        avg_train_loss_total = epoch_loss_total / steps
        avg_train_loss_small = epoch_loss_small / steps
        avg_train_loss_medium = epoch_loss_medium / steps
        epoch_time = time.time() - start_time

        avg_val_loss_total, avg_val_loss_small, avg_val_loss_medium = evaluate_model(model, val_dataset)

        print(f"✅ Treino | Total: {avg_train_loss_total:.2f} | "
              f"13x13: {avg_train_loss_small:.2f} | 26x26: {avg_train_loss_medium:.2f} || "
              f"Validação | Total: {avg_val_loss_total:.2f} | "
              f"13x13: {avg_val_loss_small:.2f} | 26x26: {avg_val_loss_medium:.2f} | "
              f"Tempo: {epoch_time:.2f}s")

        if (epoch + 1) % SAVE_FREQ == 0 or (epoch + 1) == EPOCHS:
            model_path = save_model(model, epoch+1, avg_train_loss_total, avg_val_loss_total)
            print(f"💾 Modelo salvo em: {model_path}")

    tflite_path = export_tflite(model)
    print(f"\n🚀 Modelo TFLite quantizado exportado: {tflite_path}")

if __name__ == "__main__":
    main()
