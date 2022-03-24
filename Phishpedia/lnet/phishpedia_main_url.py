from phishpedia_config import *
import os
from src.util.chrome import *
import cv2
import argparse
from lnet_main import main

if __name__ == "__main__":

    #os.environ["CUDA_VISIBLE_DEVICES"]="0"
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--folder", help='Input folder path to parse',  default='../datasets/test_sites/')
    parser.add_argument('-r', "--results", help='Input results file name', default='../test.txt')
    args = parser.parse_args()
    date = args.folder.split('/')[-1]
    directory = args.folder
    results_path = args.results.split('.txt')[0] + "_pedia.txt"

    if not os.path.exists(results_path):
        with open(results_path, "w+") as f:
            f.write("folder" + "\t")
            f.write("url" + "\t")
            f.write("phish" + "\t")
            f.write("prediction" + "\t")  # write top1 prediction only
            f.write("siamese_conf" + "\t")
            f.write("vt_result" + "\t")
            f.write("runtime" + "\n")

    while True:
        domain = input("Input a domain")
        # Remove the last screenshoted page
        os.system("rm -Rf ../datasets/test_sites/*")
        # Create a screenshot for evaluation
        os.system("python ./pyshot.py " + domain)
        for item in tqdm(os.listdir(directory)):
            start_time = time.time()

            # if item in open(results_path, encoding='ISO-8859-1').read(): # have been predicted
            #     continue

            try:
                print(item)
                full_path = os.path.join(directory, item)

                screenshot_path = os.path.join(full_path, "shot.png")
                if os.path.exists(os.path.join(full_path, 'info.txt')):
                    url = open(os.path.join(full_path, 'info.txt'), encoding='ISO-8859-1').read()
                else:
                    print("Skipped")
                    continue

                if not os.path.exists(screenshot_path):
                    continue

                else:
                    phish_category, phish_target, plotvis, siamese_conf, _, _, _ = main(url=url, screenshot_path=screenshot_path)

                    print(phish_category, phish_target)

            except Exception as e:
                raise

        #  raise(e)
        time.sleep(0.5)

