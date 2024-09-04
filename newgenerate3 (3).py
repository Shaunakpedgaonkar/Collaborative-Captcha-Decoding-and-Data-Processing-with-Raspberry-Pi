import os
import numpy as np
import random
import string
import cv2
import argparse
from captcha.image import ImageCaptcha

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--width', help='Width of captcha image', type=int)
    parser.add_argument('--height', help='Height of captcha image', type=int)
    parser.add_argument('--length', help='Length of captchas in characters', type=int)
    parser.add_argument('--count', help='How many captchas to generate', type=int)
    parser.add_argument('--output-dir', help='Where to store the generated captchas', type=str)
    parser.add_argument('--symbols', help='File with the symbols to use in captchas', type=str)
    parser.add_argument('--font-file', help='Font file to use for captchas', type=str)
    args = parser.parse_args()

    if args.width is None or args.height is None or args.length is None or args.count is None \
            or args.output_dir is None or args.symbols is None or args.font_file is None:
        print("Please provide all the required arguments")
        exit(1)

    if not os.path.exists(args.font_file):
        print("Font file not found")
        exit(1)

    captcha_generator = ImageCaptcha(fonts=[args.font_file], width=args.width, height=args.height)

    symbols_file = open(args.symbols, 'r')
    captcha_symbols = symbols_file.readline().strip()
    symbols_file.close()

    print("Generating captchas with symbol set {" + captcha_symbols + "} using font: " + args.font_file)

    if not os.path.exists(args.output_dir):
        print("Creating output directory " + args.output_dir)
        os.makedirs(args.output_dir)

    for i in range(args.count):
        random_length = random.randint(1, 6)  # Random length between 1 and 6
        random_str = ''.join([random.choice(captcha_symbols) for j in range(random_length)]) # String is created for random length
        label = random_str + '@' * (args.length - len(random_str)) # Fill the remaining characters with '@' in the label
        image_path = os.path.join(args.output_dir, label+'.png')
        if os.path.exists(image_path): # Define the image path and check for existing files with the same label
            version = 1
            while os.path.exists(os.path.join(args.output_dir, label + '_' + str(version) + '.png')):
                version += 1
            image_path = os.path.join(args.output_dir, label + '_' + str(version) + '.png')

        image = np.array(captcha_generator.generate_image(random_str))
        cv2.imwrite(image_path, image)

if __name__ == '__main__':
    main()
