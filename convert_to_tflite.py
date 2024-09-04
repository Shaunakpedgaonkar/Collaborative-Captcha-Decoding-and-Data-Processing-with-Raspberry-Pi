import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
import argparse
import tensorflow as tf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str)
    parser.add_argument('--output-dir', help='Output dir where the converted model should be saved', type=str)
    args = parser.parse_args()

    if args.model_name is None:
        print("Please specify the model to use")
        exit(1)
    
    if args.output_dir is None:
        print("Please specify the path of directory where output model should be saved")
        exit(1)

    try:
        # Load the model
        with open(f'{args.model_name}.json') as model_file:
            model_json = model_file.read()
        loaded_model = tf.keras.models.model_from_json(model_json)
        loaded_model.load_weights(f'{args.model_name}.h5')

        # Convert the model to TensorFlow Lite
        converter = tf.lite.TFLiteConverter.from_keras_model(loaded_model)
        tflite_model = converter.convert()

        # Save the converted model to a file
        with open(f'{args.output_dir}/{args.model_name}_converted.tflite', 'wb') as tflite_file:
            tflite_file.write(tflite_model)

        print(f"Conversion of {args.model_name} to TensorFlow Lite Model was successful!")
    except Exception as conversion_error:
        print(f"An error occurred during the conversion: {conversion_error}")


if __name__ == '__main__':
    main()
