import os
import tensorflow as tf
from pathlib import Path
from tensorflow.keras import layers, models
# Assuming FracOptimizer is still needed for any compilation steps
from ciphar10.optimizer import FracOptimizer 

CUSTOM_OBJECTS = {
    'FracOptimizer': FracOptimizer,
}

# Matches your provided architecture
def create_fresh_model(num_classes=10):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(32, 32, 3),
        alpha=1.0,
        include_top=False,
        weights=None, # We don't need ImageNet weights since we're loading yours
    )
    base_model.trainable = True 
    
    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

def convert_h5_to_keras():
    base_path = Path('./results/')
    h5_files = list(base_path.rglob('*.h5'))
    h5_files = [f for f in h5_files if "mobile" in str(f)]

    success_list = []
    failure_list = []

    print(f"Found {len(h5_files)} .h5 files. Starting weight injection conversion...\n")

    for h5_path in h5_files:
        keras_path = h5_path.with_suffix('.keras')
        print(f"Processing: {h5_path.name} ...", end=" ", flush=True)

        try:
            # 1. Build the exact architecture in your current Keras 2.10 environment
            model = create_fresh_model(num_classes=10)
            
            # 2. Inject weights only. This ignores the incompatible 'inbound_nodes' 
            # and 'DTypePolicy' metadata that caused previous crashes.
            model.load_weights(str(h5_path))
            
            # 3. Save as the new .keras standard
            model.save(keras_path)
            
            print(f"DONE -> {keras_path.name}")
            success_list.append(str(h5_path))

        except Exception as e:
            print(f"FAILED")
            print(f"  Error: {e}")
            failure_list.append((str(h5_path), str(e)))

    # --- FINAL REPORT ---
    print("\n" + "="*30)
    print("CONVERSION SUMMARY")
    print("="*30)
    print(f"Successfully converted: {len(success_list)}")
    print(f"Failures: {len(failure_list)}")

if __name__ == "__main__":
    convert_h5_to_keras()