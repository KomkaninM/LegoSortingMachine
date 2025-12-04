#!/usr/bin/env python3
import tensorflow as tf

SAVED_MODEL_DIR = "full_data_saved_model_tf214"
TFLITE_PATH = "full_dataset_lego.tflite"

def main():
    # Create converter from SavedModel (exported by model.export)
    converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)

    # Make it smaller & faster (good for Raspberry Pi)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    with open(TFLITE_PATH, "wb") as f:
        f.write(tflite_model)

    print("Saved TFLite model to:", TFLITE_PATH)

if __name__ == "__main__":
    main()
