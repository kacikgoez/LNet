from phishpedia_config import *
import os
import argparse
import numpy
from src.util.chrome import *
from layoutnet.util import l_eval
from PIL import Image
from io import StringIO
import pickle
import pytesseract
import json
# import os
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

#####################################################################################################################
# ** Step 1: Enter Layout detector, get predicted elements
# ** Step 2: Enter Siamese, siamese match a phishing target, get phishing target

# **         If Siamese report no target, Return Benign, None
# **         Else Siamese report a target, Return Phish, phishing target
#####################################################################################################################

buffer = StringIO()
counter = 0

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

    ####################### Step1: layout detector ##############################################
    pred_boxes, logo_conf, iboxes, iconf = pred_rcnn(im=screenshot_path, predictor=ele_model)
    pred_boxes = pred_boxes.detach().cpu().numpy()  ## get predicted logo box
    iboxes = iboxes.detach().cpu().numpy()

    pred_target, matched_coord, siamese_conf = phishpedia_classifier_logo(logo_boxes=pred_boxes,
                                                                     domain_map_path=domain_map_path,
                                                                     model=pedia_model,
                                                                     logo_feat_list=logo_feat_list,
                                                                     file_name_list=file_name_list,
                                                                     url=url,
                                                                     shot_path=screenshot_path,
                                                                     ts=siamese_ts)

    login_boxes = []
    login_area = []
    for i in iboxes:
        i_width = i[2] - i[0]
        # Cropped input field for scan
        cropInput = image.crop((i[0], i[1], i[2], i[3]))
        ocrInput = str(pytesseract.image_to_string(cropInput)).strip().casefold()
        # List of possible suspicious inputs
        possibleList = ["email", "password", "e-mail", "passwort", "username", "nutzername", "phone number", "telefonnummer"]
        # Check if detected text is in given input
        if any(pstr in ocrInput for pstr in possibleList):
            login_boxes.append(i)
            cbox_x1 = max(int(i[0] - i_width * 0.25), 0)
            cbox_y1 = max(int(i[1] - i_width * 1.25), 0)
            cbox_x2 = min(int(i[2] + i_width * 0.25), width)
            cbox_y2 = int(i[3])
            login_area.append(numpy.array([cbox_x1, cbox_y1, cbox_x2, cbox_y2]))

    login_boxes = numpy.array(login_boxes)
    login_area = numpy.array(login_area)

    ######################## Step2: Siamese (logo matcher) ########################################
    detected = lnet_phishpedia_classifier_logo(logo_boxes=pred_boxes,
                                                                     domain_map_path=domain_map_path,
                                                                     model=pedia_model,
                                                                     logo_feat_list=logo_feat_list,
                                                                     file_name_list=file_name_list,
                                                                     url=url,
                                                                     shot_path=screenshot_path,
                                                                     ts=siamese_ts)

    # Format output to vis function input
    toDraw = []
    labels = []

    # Format detected logo boxes
    for d in detected:
        l = []
        # x1, y1
        l.append(d[0])
        l.append(d[1])
        # Width, height -> x2, y2
        l.append(d[0] + d[2])
        l.append(d[1] + d[3])
        toDraw.append(l)
        # Label
        labels.append(str(d[4]) + " " + str(d[5]))

    plotvis = vis(screenshot_path, login_area, None)

    # If no element is reported
    if len(pred_boxes) == 0:
        print('No element is detected, report as benign')
        return phish_category, pred_target, plotvis, siamese_conf
    print('Entering siamese')

    #Format
    result, l_conf = l_eval(layout_model, device, [width, height], detected, debug=True)
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
        # Find two logos with highest confidence

    s_conf = numpy.array(l_conf).argsort()[::-1]
    # If conf. of second highest logo is within 0.04 and it's a brand, report
    for i in s_conf:
        if l_conf[s_conf[0]] - l_conf[i] < 0.15:
            if str(detected[result[i]][4]) != "None":
                return 1, detected[result[i]][4], plotvis, detected[result[i]][5], pred_target, siamese_conf
    for j, login_box in enumerate(login_boxes):
        above = 0
        print("Entered")
        res = None
        for k in detected:
            if within(k, login_area[j]) and str(k[4]) != "None":
                above += 1
                res = k
        if 0 < above < 3:
            return 1, res[4], plotvis, res[5], pred_target, siamese_conf
    return 0, None, plotvis, 0, pred_target, siamese_conf

def within(a, b):
    if (a[0] > b[0] and a[1] > b[1]) and (a[2] < b[2] and a[3] < b[3]):
        return True
    return False

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', "--folder", help='Input folder path to parse',  default='./datasets/test_sites')
    parser.add_argument('-r', "--results", help='Input results file name', default='./test.txt')
    args = parser.parse_args()
    date = args.folder.split('/')[-1]
    directory = args.folder
    results_path = args.results.split('.txt')[0] + "_compare.txt"
    counter = 0

    # Uncomment to scan single file for debugging purposes
    # phish_category, phish_target, plotvis, siamese_conf, pred1, conf2 = main(url="https://test.com", screenshot_path="/Volumes/Extern/holle-screenshots/screenshots/benign/00234877-screenshot.png")
    # print(phish_category, "|", phish_target, "|", siamese_conf, "|", pred1, "|", conf2)
    # cv2.imwrite(os.path.join("/Volumes/GoogleDrive/Meine Ablage/Phishpedia/lnet/layoutnet", "00234852-screenshot.png"), plotvis)
    # exit(0)

    if not os.path.exists(results_path):
        with open(results_path, "w+") as f:
            f.write("File" + "\t")
            #f.write("url" + "\t")
            f.write("LNet" + "\t")
            f.write("LNet conf." + "\t")  # write top1 prediction only
            f.write("Orig." + "\t")
            f.write("Orig. Conf" + "\n")


    # Save directory structure for Google Colab to avoid IO error
    dir_struct_name = 'holle-' + os.path.basename(os.path.normpath(directory)) + '-struct.pickle'
    if not os.path.exists(dir_struct_name):
        # Create buffer for directory items
        dirbuff = []
        # Load dir.
        dir_struct = tqdm(os.listdir(directory))
        # Add all items to list as str.
        for i in dir_struct:
            dirbuff.append(str(i))
        # Store
        with open(dir_struct_name, 'wb') as dir_file:
            pickle.dump(dirbuff, dir_file)
        # Delete buffer
        dirbuff = []
    else:
        with open(dir_struct_name, 'rb') as dir_file:
            dir_struct = pickle.load(dir_file)


    for item in dir_struct:
        start_time = time.time()

        full_path = os.path.join(directory, item)

        # if item in open(results_path, encoding='ISO-8859-1').read(): # have been predicted
        #     continue

        try:
            screenshot_path = os.path.join(directory, item)
            #url = open(os.path.join(full_path, 'info.txt'), encoding='ISO-8859-1').read()
            url = "https://www.test-on-holle.com"

            if 0 == 1:
                continue
            else:
                phish_category, phish_target, plotvis, siamese_conf, pred_target_2, siamese_conf_2 = main(url=url, screenshot_path=screenshot_path)
                counter += 1

                ''''# FIXME: call VTScan only when phishpedia report it as phishing
                vt_result = "None"
                if 0 == 1:
                    try:
                        if vt_scan(url) is not None:
                            positive, total = vt_scan(url)
                            print("Positive VT scan!")
                            vt_result = str(positive) + "/" + str(total)
                        else:
                            print("Negative VT scan!")
                            vt_result = "None"

                    except Exception as e:
                        print('VTScan is not working...')
                        vt_result = "error"'''

                vt_result = "None"

                # write results as well as predicted images
                buffer.write(item + "\t")  # write top1 prediction only
                buffer.write(str(phish_target) + "\t")  # write top1 prediction only
                buffer.write(str(siamese_conf) + "\t")
                buffer.write(str(pred_target_2) + "\t")  # write top1 prediction only
                buffer.write(str(siamese_conf_2) + "\n")  # write top1 prediction only

                if counter > 25:
                    with open(results_path, "a+", encoding='ISO-8859-1') as f:
                        f.write(buffer.getvalue())
                    counter = 0
                    buffer = StringIO()

                #cv2.imwrite(os.path.join(full_path, "./local_predict.png"), plotvis)
                #cv2.imwrite(os.path.join(full_path, "predict.png"), plotvis)

        except Exception as e:
            raise




