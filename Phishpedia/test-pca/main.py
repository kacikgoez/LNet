import sys
sys.path.append('../')

from config import *
import os
import glob
from PIL import Image
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

'''
Executing this file test the PCA version of Phishpedia and the original version. The entire expanded target list that 
is under src > siamese will be evaluated, and the mean confidence error will be calculated.
'''

def main(path):
    '''
    Phishdiscovery for phishpedia main script
    :param url: URL
    :param screenshot_path: path to screenshot
    :return phish_category: 0 for benign 1 for phish
    :return pred_target: None or phishing target
    :return plotvis: predicted image
    :return siamese_conf: siamese matching confidence
    '''

    mean_error = 0
    counter = 0

    path_target_list = path

    print("Eval PCA for all logos in the database")

    for dir_path in os.listdir(path_target_list):
        parent_path = os.path.join(path_target_list, dir_path + "/")
        for i in glob.glob(parent_path + "[0-9]*.png"):
            screenshot_path = str(i)
            url = "https://www.test.com"
            im = Image.open(screenshot_path)
            width, height = im.size
            pred_boxes = [[0, 0, width, height]]
            ######################## Step2: Siamese (logo matcher) ########################################
            pca_target, matched_coord2, siamese_conf2 = phishpedia_classifier_logo(pca, logo_boxes=pred_boxes,
                                                                             domain_map_path=domain_map_path,
                                                                             model=pedia_model,
                                                                             logo_feat_list=logo_feat_list,
                                                                             file_name_list=file_name_list,
                                                                             url=url,
                                                                             shot_path=screenshot_path,
                                                                             ts=siamese_ts)

            pedia_target, matched_coord, siamese_conf = phishpedia_classifier_logo(None, logo_boxes=pred_boxes,
                                                                             domain_map_path=domain_map_path,
                                                                             model=pedia_model,
                                                                             logo_feat_list=logo_feat_list2,
                                                                             file_name_list=file_name_list2,
                                                                             url=url,
                                                                             shot_path=screenshot_path,
                                                                             ts=siamese_ts)
            if siamese_conf2 is not None and siamese_conf is not None:
                error = siamese_conf2 - siamese_conf
                mean_error += error
                counter += 1
                if abs(error) > 0.1:
                    print("Deviation of " + str(i) + ":", error)
            else:
                print("None detected!")
                print(siamese_conf2, siamese_conf)

    print(mean_error , counter)
    print("Average error: ", (mean_error / counter))

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    main("../src/siamese_pedia/expand_targetlist/")

'''if __name__ == "__main__":

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
        os.system("rm -R ../datasets/test_sites/*")
        # Create a screenshot for evaluation
        os.system("python pyshot.py " + domain)
        print(directory)
        print(os.listdir(directory))
        print(directory)
        for item in tqdm(os.listdir(directory)):
            start_time = time.time()

            # if item in open(results_path, encoding='ISO-8859-1').read(): # have been predicted
            #     continue

            try:
                print(item)
                full_path = os.path.join(directory, item)

                screenshot_path = os.path.join(full_path, "shot.png")
                url = open(os.path.join(full_path, 'info.txt'), encoding='ISO-8859-1').read()

                if not os.path.exists(screenshot_path):
                    continue

                else:
                    phish_category, phish_target, plotvis, siamese_conf = main(url=url, screenshot_path=screenshot_path)

                    print(phish_category, phish_target, siamese_conf)

                    # FIXME: call VTScan only when phishpedia report it as phishing
                    vt_result = "None"
                    if phish_target is not None:
                        try:
                            if vt_scan(url) is not None:
                                positive, total = vt_scan(url)
                                print("Positive VT scan!")
                                vt_result = str(positive) + "/" + str(total)
                            else:
                                print("Negative VT scan!")
                                vt_result = "None"

                        except Exception as e:
                            print(e)
                            print('VTScan is not working...')
                            vt_result = "error"

                    # write results as well as predicted images
                    with open(results_path, "a+", encoding='ISO-8859-1') as f:
                        f.write(item + "\t")
                        f.write(url + "\t")
                        f.write(str(phish_category) + "\t")
                        f.write(str(phish_target) + "\t")  # write top1 prediction only
                        f.write(str(siamese_conf) + "\t")
                        f.write(vt_result + "\t")
                        f.write(str(round(time.time() - start_time, 4)) + "\n")

                    cv2.imwrite(os.path.join(full_path, "predict.png"), plotvis)

            except Exception as e:
                raise

        #  raise(e)
        time.sleep(0.5)'''



