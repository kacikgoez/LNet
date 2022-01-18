from phishpedia_config import *
import os
import argparse
import time
from src.util.chrome import *
from layoutnet.util import l_eval
from PIL import Image
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#####################################################################################################################
# ** Step 1: Enter Layout detector, get predicted elements
# ** Step 2: Enter Siamese, siamese match a phishing target, get phishing target

# **         If Siamese report no target, Return Benign, None
# **         Else Siamese report a target, Return Phish, phishing target
#####################################################################################################################


def main(url, screenshot_path):
    '''
    Phishdiscovery for phishpedia main script
    :param url: URL
    :param screenshot_path: path to screenshot
    :return phish_category: 0 for benign 1 for phish
    :return pred_target: None or phishing target
    :return plotvis: predicted image
    :return siamese_conf: siamese matching confidence
    '''

    # 0 for benign, 1 for phish, default is benign
    phish_category = 0
    pred_target = None
    siamese_conf = None
    print("Entering phishpedia")

    image = Image.open(screenshot_path)
    width, height = image.size

    current = time.time()
    ####################### Step1: layout detector ##############################################
    pred_boxes, logo_conf, iboxes, iconf = pred_rcnn(im=screenshot_path, predictor=ele_model)
    pred_boxes = pred_boxes.detach().cpu().numpy()  ## get predicted logo box
    pred_box_time = time.time()
    print("PRED RCNN TIME:", pred_box_time - current)

    print("PRED", pred_boxes)


    current = time.time()
    ######################## Step2: Siamese (logo matcher) ########################################
    detected = lnet_phishpedia_classifier_logo(logo_boxes=pred_boxes,
                                                                     domain_map_path=domain_map_path,
                                                                     model=pedia_model,
                                                                     logo_feat_list=logo_feat_list,
                                                                     file_name_list=file_name_list,
                                                                     url=url,
                                                                     shot_path=screenshot_path,
                                                                     ts=siamese_ts)
    print("LNET TIME:", time.time() - current)
    # Format output to vis function input
    toDraw = []
    labels = []
    for d in detected:
        l = []
        l.append(d[0])
        l.append(d[1])
        l.append(d[0] + d[2])
        l.append(d[1] + d[3])
        toDraw.append(l)
        labels.append(str(d[4]) + " " + str(d[5]))
    print(toDraw, labels)
    #plotvis = vis(screenshot_path, iboxes, labels)
    plotvis = vis(screenshot_path, iboxes, None)

    print(detected)

    # If no element is reported
    if len(pred_boxes) == 0:
        print('No element is detected, report as benign')
        return phish_category, pred_target, plotvis, siamese_conf
    print('Entering siamese')

    #Format
    result, _ = l_eval(layout_model, device, [width, height], detected)
    print(result)

    ''' if pred_target is None:
        print('Did not match to any brand, report as benign')
        return phish_category, pred_target, plotvis, siamese_conf

    else:
        phish_category = 1
        # Visualize, add annotations
        cv2.putText(plotvis, "Target: {} with confidence {:.4f}".format(pred_target, siamese_conf),
                    (int(matched_coord[0] + 20), int(matched_coord[1] + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)'''

    #return phish_category, pred_target, plotvis, siamese_conf
    if len(result) != 0 and str(detected[result[0]][4]) != "None":
        print("DETECTED PHISHING PAGE: " + str(detected[result[0]][4]))
        return 1, detected[result[0]][4], plotvis, detected[result[0]][5]
    else:
        #print("NO PHISHING PAGE DETECTED")
        return 0, None, plotvis, 0


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
        os.system("rm -R ../datasets/test_sites/*")
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
                url = open(os.path.join(full_path, 'info.txt'), encoding='ISO-8859-1').read()

                if not os.path.exists(screenshot_path):
                    continue

                else:
                    phish_category, phish_target, plotvis, siamese_conf = main(url=url, screenshot_path=screenshot_path)

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
                    '''with open(results_path, "a+", encoding='ISO-8859-1') as f:
                        f.write(item + "\t")
                        f.write(url + "\t")
                        f.write(str(phish_category) + "\t")
                        f.write(str(phish_target) + "\t")  # write top1 prediction only
                        f.write(str(siamese_conf) + "\t")
                        f.write(vt_result + "\t")
                        f.write(str(round(time.time() - start_time, 4)) + "\n")'''

                    cv2.imwrite(os.path.join(full_path, "predict.png"), plotvis)

            except Exception as e:
                raise

        #  raise(e)
        time.sleep(0.5)

