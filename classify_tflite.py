import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import numpy

import argparse
import tflite_runtime.interpreter as tflite 
from PIL import Image


def decode(characters, y):
    y = numpy.argmax(numpy.array(y), axis=1)
    return ''.join([characters[x] for x in y])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', help='Model name to use for classification', type=str)
    parser.add_argument('--captcha-dir', help='Where to read the captchas to break', type=str)
    parser.add_argument('--output', help='File where the classifications should be saved', type=str)
    parser.add_argument('--symbols', help='File with the symbols to use in captchas', type=str)
    args = parser.parse_args()

    if args.model_name is None:
        print("Please specify the CNN model to use")
        exit(1)

    if args.captcha_dir is None:
        print("Please specify the directory with captchas to break")
        exit(1)

    if args.output is None:
        print("Please specify the path to the output file")
        exit(1)

    if args.symbols is None:
        print("Please specify the captcha symbols file")
        exit(1)

    symbols_file = open(args.symbols, 'r')
    captcha_symbols = symbols_file.readline().strip()
    symbols_file.close()

    print("Classifying captchas with symbol set {" + captcha_symbols + "}")

    img_files = os.listdir(args.captcha_dir)
    img_files = sorted(img_files)


    with open(args.output, 'w',newline='\n') as output_file:
        
        interpreter = tflite.Interpreter(args.model_name+'.tflite')
        interpreter.allocate_tensors()


        for x in img_files:
            # load image and preprocess it
            raw_data = Image.open(os.path.join(args.captcha_dir, x))
            image = numpy.array(raw_data) / 255.0
            (c, h, w) = image.shape
            image = image.reshape([-1, c, h, w])

            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            image = image.astype('float32')
            interpreter.set_tensor(input_details[0]['index'],image)
            interpreter.invoke()
            captcha_predicted = ""
            print(output_details)
            output_0 = interpreter.get_tensor(output_details[3]['index'])
            output_1 = interpreter.get_tensor(output_details[5]['index'])
            output_2 = interpreter.get_tensor(output_details[0]['index'])
            output_3 = interpreter.get_tensor(output_details[4]['index'])
            output_4 = interpreter.get_tensor(output_details[2]['index'])
            output_5 = interpreter.get_tensor(output_details[1]['index'])

            output_0 = decode(captcha_symbols, output_0)
            output_1 = decode(captcha_symbols, output_1)
            output_2 = decode(captcha_symbols, output_2)
            output_3 = decode(captcha_symbols, output_3)
            output_4 = decode(captcha_symbols, output_4)
            output_5 = decode(captcha_symbols, output_5)
            captcha_predicted = captcha_predicted + output_0 + output_1 + output_2 + output_3 + output_4 + output_5

            output_file.write(x + "," + captcha_predicted + "\n")

            print('Classified ' + x)

if __name__ == '__main__':
    main()
