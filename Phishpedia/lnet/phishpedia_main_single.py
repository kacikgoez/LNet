import os
import cv2
from lnet_main import main

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    counter = 0

    # Uncomment to scan single file for debugging purposes
    default_path = r"/Volumes/GoogleDrive/Meine Ablage/holle2/screenshots/benign"

    while True:
        if default_path != "":
            file_input = input("File path").strip()
        else:
            print("Warning! Default path is to: " + default_path)
            file_input = input("File name").strip()
        phish_category, phish_target, plotvis, siamese_conf, pred1, conf2, lnet_conf = main(url="https://test.com", screenshot_path=os.path.join(default_path, file_input), save=True)
        print(phish_category, "|", phish_target, "|", siamese_conf, "|", pred1, "|", conf2)
        if plotvis != None:
            cv2.imwrite("./single-screenshot.png", plotvis)