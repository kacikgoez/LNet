from phishpedia_config import *
import os
import numpy
from src.util.chrome import *
from layoutnet.util import l_eval
from PIL import Image
import cv2

# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

#####################################################################################################################
# ** Step 1: Enter Layout detector, get predicted elements
# ** Step 2: Enter Siamese, siamese match a phishing target, get phishing target

# **         If Siamese report no target, Return Benign, None
# **         Else Siamese report a target, Return Phish, phishing target
#####################################################################################################################

def main(url, screenshot_path, save=False):
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

    screenshot = cv2.imread(r"" + screenshot_path)

    ####################### Step1: layout detector ##############################################
    button_boxes, button_conf, info_conf, info_boxes, nav_boxes, nav_conf, popup_boxes, popup_conf = nav_rcnn(im=screenshot, predictor=nav_model)

    '''# Crop out info box to adjust
    if info_boxes.numel() > 0:
        if info_boxes[0][1] < 50:
            h, w = screenshot.shape[:2]
            # Crop out info box, but leave 50px of it incase the network includes the navbar in it
            screenshot = screenshot[max(int(info_boxes[0][3]) - 50, 0) : h, 0 : w]'''

    height, width = screenshot.shape[:2]

    pred_boxes, logo_conf, iboxes, iconf = pred_rcnn(im=screenshot, predictor=ele_model)
    pred_boxes = pred_boxes.detach().cpu().numpy()  ## get predicted logo box
    iconf = iconf.detach().cpu().numpy()

    logo_conf = logo_conf.detach().cpu().numpy()

    print("LCONF", logo_conf )

    if len(iconf) > 0:
        iboxes = iboxes[numpy.argwhere(iconf >= 0.65).flatten()]

    screenshot = cv2.cvtColor(screenshot, cv2.COLOR_BGR2RGB)
    screenshot = Image.fromarray(screenshot)

    pred_boxes_orig = []

    for i, v in enumerate(logo_conf):
        if v >= 0.05:
            pred_boxes_orig.append(pred_boxes[i])

    pred_boxes_orig = numpy.array(pred_boxes)

    '''lnet_boxes = []
    for i, conf in enumerate(logo_conf):
        if conf >= 0.1:
            lnet_boxes.append(pred_boxes[i])'''

    print("Logo Conf", logo_conf)

    iboxes = iboxes.detach().cpu().numpy()

    pred_target, matched_coord, siamese_conf = phishpedia_classifier_logo(logo_boxes=pred_boxes_orig,
                                                                     domain_map_path=domain_map_path,
                                                                     model=pedia_model,
                                                                     logo_feat_list=logo_feat_list,
                                                                     file_name_list=file_name_list,
                                                                     url=url,
                                                                     shot_path=screenshot_path,
                                                                     ts=siamese_ts)

    login_boxes = []
    login_area = []
    # Iterate through input boxes
    start = time.time()
    for i in iboxes:
        i_width = i[2] - i[0]
        # Factor for local neighbourhood
        factor = (i[3] - i[1])
        if width / factor <= 3:
            break
        # Cropped input field and local neighbourhood
        # List of possible suspicious inputs
        '''possibleList = ["email", "@", "password", "e-mail", "passwort", "id", "username", "nutzername", "telefon", "phone number", "phone", "mobil",
                        "telefonnummer", "имя пользователя", "электронная почта"]'''
        # Check if detected text is in given input
        '''if any(pstr.casefold() in ocrOutput.casefold() for pstr in possibleList):
            login_boxes.append(i)'''
        login_boxes.append(i)
        cbox_x1 = max(int(i[0] - max(i_width * 0.25, 30)), 0)
        cbox_y1 = max(int(i[1] - max(i_width * 0.7, 200)), 0)
        cbox_x2 = min(int(i[2] + max(i_width * 0.25, 30)), width)
        cbox_y2 = int(i[1])
        login_area.append(numpy.array([cbox_x1, cbox_y1, cbox_x2, cbox_y2]))
    print("Input Scan", time.time() - start, len(iboxes))

    login_boxes = numpy.array(login_boxes)
    login_area = numpy.array(login_area)

    pred_boxes = pred_boxes
    ######################## Step2: Siamese (logo matcher) ########################################
    detected, ops_boxes = lnet_phishpedia_classifier_logo(logo_boxes=pred_boxes,
                                                                     logo_conf=logo_conf,
                                                                     domain_map_path=domain_map_path,
                                                                     model=pedia_model,
                                                                     logo_feat_list=logo_feat_list,
                                                                     file_name_list=file_name_list,
                                                                     word_list=word_list,
                                                                     url=url,
                                                                     shot_path=screenshot,
                                                                     keras_pipeline=keras_pipeline,
                                                                     ts=lnet_siamese_ts,
                                                                    debug=True)

    print("detected", detected)
    temp_logo = []
    for i, j in enumerate(detected):
        if logo_conf[0] - j[6] < 0.42 and str(j[4]) == "None":
            temp_logo.append(j)
        elif str(j[4]) != "None":
            temp_logo.append(j)
    detected = temp_logo

    # If info box (cookie box for example) at the top shifts page, remove logos within and shift

    keep = []
    if info_conf.numel() > 0:
        if info_conf[0] >= 0.7 and info_boxes[0][1] <= 50 and info_boxes[0][3] - info_boxes[0][1] >= 200:
            for j in range(len(detected)):
                if int(info_boxes[0][1] * 0.75) > detected[j][1]:
                    pass
                else:
                    detected[j][1] -= int(info_boxes[0][3] * 0.75)
                    keep.append(detected[j])
            detected = keep

    print(detected)

    print("BUTTON CO", button_conf)


    print(button_conf)
    for j, btn in enumerate(button_boxes):
        for i, k in enumerate(detected):
            # Possible sign in options to remove, only remove if ratio of 2.5
            if button_conf[j] >= 0.9 and 4 <= (btn[2] - btn[0]) / (btn[3] - btn[1]) <= 10:
                 if within(boxCenter(k), btn):
                    del detected[i]

    # Format output to vis function input
    toDraw = []
    labels = []

    '''# Format detected logo boxes
    for d in detected:
        l = []
        # x1, y1
        l.append(int(d[0]))
        l.append(int(d[1]))
        # Width, height -> x2, y2
        l.append(int(d[0] + d[2]))
        l.append(int(d[1] + d[3]))
        toDraw.append(l)
        # Label
        #labels.append(str(d[4]) + " " + str(d[5]))'''

    # If no element is reported
    plotvis = None
    if len(pred_boxes) == 0:
        print('No element is detected, report as benign')
        return phish_category, pred_target, plotvis, siamese_conf, None, None, 0
    print('Entering siamese')

    plotvis = None
    if save:
        try:
            plotvis = vis(screenshot_path, torch.tensor(ops_boxes), None)
            cv2.imwrite("./single-screenshot.png", plotvis)
        except IndexError:
            return 0, 0, 0, 0, 0, 0, 0

    #Format 00234334-screenshot.png
    result, l_conf = l_eval(layout_model, layout_model_7, device, [width, height], detected, debug=True)

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

    plotvis = None
    s_conf = numpy.array(l_conf).argsort()[::-1]
    # Consider compositions of every logo
    maxNone = 0
    for i in s_conf:
        if detected[result[i]][4] != "None" and maxNone != 0:
            maxNone = l_conf[i]
        if l_conf[s_conf[0]] - l_conf[s_conf[i]] < 0.4:
            if str(detected[result[s_conf[i]]][4]) != "None":
                return 1, detected[result[s_conf[i]]][4], plotvis, detected[result[i]][5], pred_target, siamese_conf, l_conf[s_conf[i]]
    # Only consider navbar
    nav = []
    # Only eval. if navbox is found
    print("NAVCONF", nav_conf)
    if nav_boxes.numel() != 0:
        for i, _ in enumerate(nav_boxes):
            if nav_conf[i] >= 0.85:
                for k in detected:
                    if within(boxCenter(k), nav_boxes[i]):
                        nav.append(k)
        nav_result, nav_l_conf = l_eval(layout_model, layout_model_7, device, [width, height], nav, debug=True)
        # Sort by confidence (indeces of original)
        nav_s_conf = numpy.array(nav_l_conf).argsort()[::-1]
        for i in nav_s_conf:
            if nav_l_conf[nav_s_conf[i]] >= 0.05 and nav_l_conf[nav_s_conf[0]] - nav_l_conf[nav_s_conf[i]] < 0.15:
                # If brand is not none, return
                if str(nav[nav_result[nav_s_conf[i]]][4]) != "None":
                    return 1, detected[nav_result[nav_s_conf[i]]][4], plotvis, detected[nav_result[i]][5], pred_target, siamese_conf, nav_l_conf[nav_s_conf[i]]
    for j, login_box in enumerate(login_boxes):
        above = 0
        res = None
        for k in detected:
            if within(boxCenter(k), login_area[j]) and str(k[4]) != "None":
                above += 1
                res = k
        if 0 < above < 3:
            print("login")
            return 1, res[4], plotvis, res[5], pred_target, siamese_conf, -1
    return 0, None, plotvis, 0, pred_target, siamese_conf, maxNone

# Check if A is within boundary B
def within(a, b):
    if (a[0] > b[0] and a[1] > b[1]) and (a[2] < b[2] and a[3] < b[3]):
        return True
    return False

def boxCenter(box):
    centerX = int(box[0] + (box[2] / 2))
    centerY = int(box[1] + (box[3] / 2))
    return [centerX, centerY, centerX, centerY]
