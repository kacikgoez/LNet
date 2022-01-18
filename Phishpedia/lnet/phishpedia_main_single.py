from phishpedia_config import *
import os
import argparse
import numpy
from src.util.chrome import *
from layoutnet.util import l_eval
from PIL import Image
from io import StringIO
import pickle
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
    pred_boxes, logo_conf, iboxes, _ = pred_rcnn(im=screenshot_path, predictor=ele_model)
    pred_boxes = pred_boxes.detach().cpu().numpy()  ## get predicted logo box

    # Heuristic rule for testing purposes
    ibox = None
    for i in iboxes:
        i_width = i[2] - i[0]
        # Crop box
        cbox_x1 = max(int(i[0] - i_width * 0.15), 0)
        cbox_y1 = max(int(i[1] - i_width * 1.15), 0)
        cbox_x2 = min(int(i[2] + i_width * 0.15), width)
        cbox_y2 = min(int(i[3] + i_width * 0.15), width)
        ibox = numpy.array([[cbox_x1, cbox_y1, cbox_x2, cbox_y2]])

    if len(pred_boxes) == 0:
        print('No element is detected, report as benign')
        return 0, None, [], 0, None, 0
    print('Entering siamese')

    ######################## Step2: Siamese (logo matcher) ########################################
    detected = lnet_phishpedia_classifier_logo(logo_boxes=pred_boxes,
                                               domain_map_path=domain_map_path,
                                               model=pedia_model,
                                               logo_feat_list=logo_feat_list,
                                               file_name_list=file_name_list,
                                               url=url,
                                               shot_path=screenshot_path,
                                               ts=siamese_ts)

    pred_target, matched_coord, siamese_conf = phishpedia_classifier_logo(logo_boxes=pred_boxes,
                                                                     domain_map_path=domain_map_path,
                                                                     model=pedia_model,
                                                                     logo_feat_list=logo_feat_list,
                                                                     file_name_list=file_name_list,
                                                                     url=url,
                                                                     shot_path=screenshot_path,
                                                                     ts=siamese_ts)

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

    # Draw the predictions
    plotvis = vis(screenshot_path, ibox, None)
    #plotvis = vis(screenshot_path, iboxes, None)

    # If no element is reported

    # Format
    result, l_conf = l_eval(layout_model, device, [width, height], detected, debug=True)

    ''' if pred_target is None:
        print('Did not match to any brand, report as benign')
        return phish_category, pred_target, plotvis, siamese_conf

    else:
        phish_category = 1
        # Visualize, add annotations
        cv2.putText(plotvis, "Target: {} with confidence {:.4f}".format(pred_target, siamese_conf),
                    (int(matched_coord[0] + 20), int(matched_coord[1] + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)'''

    if len(result) > 1:
        # Find two logos with highest confidence
        s_conf = numpy.array(l_conf).argsort()[::-1]
        # Heuristic rule to determine if further steps are neccessary
        '''if dist > 0.08:
            return 1, detected[result[0]][4], plotvis, detected[result[0]][5], pred_target, siamese_conf
        # Areas of top 2 results
        rs1 = detected[result[s_conf[0]]][2] * detected[result[s_conf[0]]][3]
        rs2 = detected[result[s_conf[1]]][2] * detected[result[s_conf[1]]][3]
        # If one is sig. larger, count that one
        if (rs1 / rs2) > 1.6 and dist <= 0.08:
            return 1, detected[result[0]][4], plotvis, detected[result[0]][5], pred_target, siamese_conf
        else:
            return 0, "UNCERTAIN", plotvis, 0, pred_target, siamese_conf'''

        # If conf. of second highest logo is within 0.04 and it's a brand, report
        for i in s_conf:
            if l_conf[s_conf[0]] - l_conf[i] < 0.15:
                if str(detected[result[i]][4]) != "None":
                    return 1, detected[result[i]][4], plotvis, detected[result[i]][5], pred_target, siamese_conf

        for i in s_conf:
            if within(detected[result[i]], ibox) and str(detected[result[i]][4]) != "None":
                return 1, detected[result[i]][4], plotvis, detected[result[i]][5], pred_target, siamese_conf
    elif len(result) == 1:
        return 1, detected[result[0]][4], plotvis, detected[result[0]][5], pred_target, siamese_conf

    return 0, None, plotvis, 0, pred_target, siamese_conf


# A within B, returns bool
def within(a, b):
    ac = [(a[2] - a[0]), (a[3] - a[1])]
    if (ac[0] > b[0] and ac[0] < b[2]) and (ac[1] > b[1] and ac[1] < b[3]):
        return True
    return False


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    counter = 0

    # Uncomment to scan single file for debugging purposes
    default_path = input("Any default path? Leave empty if none")

    while True:
        if default_path != "":
            file_input = input("File path")
        else:
            file_input = input("File name")
        phish_category, phish_target, plotvis, siamese_conf, pred1, conf2 = main(url="https://test.com", screenshot_path=os.path.join(default_path, file_input))
        print(phish_category, "|", phish_target, "|", siamese_conf, "|", pred1, "|", conf2)
        cv2.imwrite(os.path.join("/Volumes/GoogleDrive/Meine Ablage/Phishpedia/lnet/", "single-screenshot.png"), plotvis)