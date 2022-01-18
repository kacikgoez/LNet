import torch
import numpy
from .WebLayoutData import direct_normalize_layout, calculate_bin, channel_of_logo
from .main import WebLayoutNet

# Times 1.2 since Phishpedia RPN doesnt propose exact boxes
default_sizes = numpy.array([13, 53, 1023]) * 1.2
'''
   Takes in the image resolution and bounding boxes.
   res = [width, height]
   bbox = [[x, y, width, height], ...]
   returns bounding boxes normalized
'''

def format_to_layoutnet(res, bboxes):
    width = res[0]
    height = res[1]
    boxes = []
    for i in bboxes:
        box = dict()
        box["x"] = (i[0] / width) * 100
        box["y"] = (i[1] / height) * 100
        box["width"] = (i[2] / width) * 100
        box["height"] = (i[3] / height) * 100
        boxes.append(box)
    return boxes

'''
    Loads the layout model
'''
def load_model():
    return torch.load("./f-model.pth")

'''
    Evalutes each box:
    model - from load_model
    all_boxes = [[x,y,width,height],...]
'''
def l_eval(model, device, res, all_boxes, debug=False):
    positive = []
    conf = []
    grid = 7
    cSep = 3
    boxes = format_to_layoutnet(res, all_boxes)
    norm = direct_normalize_layout(boxes, screen_res=res[0])

    if len(boxes) > 0:
        bins = []
        for i in boxes:
            bin = calculate_bin(grid, norm, i)
            bins.append(bin)
        for i, iv in enumerate(boxes):
            p = numpy.zeros((cSep * 2, grid, grid), dtype=int)
            # Set to identity logo
            p[channel_of_logo(cSep, default_sizes, iv["width"] * iv["height"], 0)][bins[i][0]][bins[i][1]] += 1
            for j, jv in enumerate(boxes):
                # Set to fake logo
                if j != i:
                    '''print(jv["width"] * jv["height"])'''
                    p[channel_of_logo(cSep, default_sizes, jv["width"] * jv["height"], 1)][bins[j][0]][bins[j][1]] += 1
            if device == 'cuda':
                bin_box = torch.tensor(p).float().unsqueeze(0).unsqueeze(0).cuda()
            else:
                bin_box = torch.tensor(p).float().unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                predict = model.soft_forward(bin_box)
                if debug is True:
                    print(bin_box)
                    print(predict)
                    print("---------------------------------")
                # Scale required confidence depending on the number of logos detected
                if predict[0][1] > min(0.39 + 0.12 * len(all_boxes), 0.87):
                    positive.append(i)
                    conf.append(predict[0][1])
        return positive, conf
    else:
        return [], []

