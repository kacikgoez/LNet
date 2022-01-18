import torch
from torch.utils.data import Dataset
import json
import numpy
import random

container_size = 1200

# Diff. example 00175989-screenshot.png
class WebData(Dataset):

    '''
        grid: resolution of the histrogram
        dataJson: path to json file that contains labeled data
        sep: number of size separation channels
        expand: add random alterations (data augmentation)
        expandOnly: only return augmented data, original dataset is removed
    '''
    def __init__(self, grid, dataJson, sep = 3, expand=True, expandOnly = False):
        # Number of different sizes
        self.cSep = sep
        # Channel size for array, 2 channels for each size difference (logo, not logo)
        self.aSize = self.cSep * 2
        # Grid resolution
        self.grid = grid

        self.logo_bin = []

        self.logo_area = {}

        self.pos_logo_area = []
        self.neg_logo_area = []

        self.logo_uri = []
        self.correct_logo_area = []

        self.logo_percentile = []
        self.logo_class = []
        self.div = 100 / grid

        # Bootstrap (or other CSS frameworks) center content with a max-width
        # this can change the layout. Normalization step required.

        # Problem: 00183598-screenshot.png
        with open(dataJson, "r") as dataFile:
            webdata = json.load(dataFile)
            for i, entry in enumerate(webdata):
                for j, label in enumerate(webdata[i]["annotations"][0]["result"]):
                    values = label["value"]
                    try:
                        if values["rectanglelabels"][0] != 0:
                            self.logo_area[label["id"]] = values["width"] * values["height"]
                    except Exception as e:
                        webdata[i]["annotations"][0]["result"].pop(j)

            for i in range(self.cSep):
                # Add cut off for 33.3% and 66.6%
                self.logo_percentile.append(numpy.percentile(numpy.array(list(self.logo_area.values())), (100/self.cSep) * (i+1), axis=0))
            print(self.logo_percentile)

            if not expandOnly:
                for i in range(1):
                    for i, entry in enumerate(webdata):
                        logos = []
                        resolution = normalize_layout(webdata[i]["annotations"][0]["result"])
                        array = numpy.zeros((self.aSize, grid, grid), dtype=int)
                        # Add right combinations
                        for j, label in enumerate(webdata[i]["annotations"][0]["result"]):
                            values = label["value"]
                            bins = calculate_bin(self.grid, resolution, values)
                            if values["rectanglelabels"][0] == "logo":
                                logos.append([channel_of_logo(self.cSep, self.logo_percentile, self.logo_area[label["id"]], 0), bins[0], bins[1]])
                            else:
                                array[channel_of_logo(self.cSep, self.logo_percentile, self.logo_area[label["id"]]), bins[0], bins[1]] += 1
                        narr = torch.from_numpy(self.pick_random(logos, array))
                        if torch.count_nonzero(narr[0:3]) > 0:
                            self.logo_bin.append(narr)
                            self.logo_class.append(1)
                            self.logo_uri.append("file:///Volumes/Extern/holle-screenshots/screenshots/benign/" + webdata[i]["data"]["image"].split("benign/")[1])

                    for i in range(3):
                        for i, entry in enumerate(webdata):
                            not_logos = []
                            resolution = normalize_layout(webdata[i]["annotations"][0]["result"])
                            array = numpy.zeros((self.aSize, grid, grid), dtype=int)
                            # Add right combinations
                            for j, label in enumerate(webdata[i]["annotations"][0]["result"]):
                                values = label["value"]
                                # Calculate what bin it belongs to
                                bins = calculate_bin(grid, resolution, values)
                                if values["rectanglelabels"][0] == "logo":
                                    array[channel_of_logo(self.cSep, self.logo_percentile, self.logo_area[label["id"]]), bins[0], bins[1]] += 1
                                else:
                                    array[channel_of_logo(self.cSep, self.logo_percentile, self.logo_area[label["id"]]), bins[0], bins[1]] += 1
                                    not_logos.append([channel_of_logo(self.cSep, self.logo_percentile, self.logo_area[label["id"]], 0), bins[0], bins[1]])
                            rand = self.pick_random(not_logos, array, True)
                            narr = torch.from_numpy(rand)
                            if torch.count_nonzero(narr[0:3]) > 0:
                                self.logo_bin.append(narr)
                                self.logo_class.append(0)
                                self.logo_uri.append("file:///Volumes/Extern/holle-screenshots/screenshots/benign/" + webdata[i]["data"]["image"].split("benign/")[1])

            if expandOnly or expand:
                distinct = []
                for i, entry in enumerate(webdata):
                    logos = []
                    resolution = normalize_layout(webdata[i]["annotations"][0]["result"])
                    array = numpy.zeros((self.aSize, grid, grid), dtype=int)
                    # Add right combinations
                    for j, label in enumerate(webdata[i]["annotations"][0]["result"]):
                        values = label["value"]
                        # Calculate what bin it belongs to
                        bins = calculate_bin(grid, resolution, values)
                        if values["rectanglelabels"][0] == "logo":
                            logos.append([channel_of_logo(self.cSep, self.logo_percentile, self.logo_area[label["id"]], 0), bins[0], bins[1]])
                    narr = torch.from_numpy(self.pick_random(logos, array))
                    if torch.count_nonzero(narr[0:3]) > 0:
                        distinct.append(str(narr))
                        self.logo_bin.append(narr)
                        self.logo_class.append(1)
                        self.logo_uri.append("file:///Volumes/Extern/holle-screenshots/screenshots/benign/" +
                                             webdata[i]["data"]["image"].split("benign/")[1])
                for r in range(4):
                    for i, entry in enumerate(webdata):
                        logos = []
                        resolution = normalize_layout(webdata[i]["annotations"][0]["result"])
                        array = numpy.zeros((self.aSize, grid, grid), dtype=int)
                        # Add right combinations
                        for j, label in enumerate(webdata[i]["annotations"][0]["result"]):
                            values = label["value"]
                            # Calculate what bin it belongs to
                            bins = calculate_bin(grid, resolution, values)
                            if values["rectanglelabels"][0] != "logo":
                                logos.append([channel_of_logo(self.cSep, self.logo_percentile, self.logo_area[label["id"]], 0), bins[0], bins[1]])
                        narr = torch.from_numpy(self.pick_random(logos, array))
                        if torch.count_nonzero(narr[0:3]) > 0 and str(narr) not in distinct:
                            self.logo_bin.append(narr)
                            self.logo_class.append(0)
                            self.logo_uri.append("file:///Volumes/Extern/holle-screenshots/screenshots/benign/" +
                                                 webdata[i]["data"]["image"].split("benign/")[1])

        self.classes = ["logo", "not logo"]
        self.class_to_label = {c: i for i, c in enumerate(self.classes)}
        print("Positive", self.logo_class.count(1))
        print("Negative", self.logo_class.count(0))

    def __getitem__(self, index):
        return self.logo_bin[index].unsqueeze(0), self.logo_class[index], self.logo_uri[index]

    def __len__(self):
        return len(self.logo_bin)

    # pick random logo, revertOne = +1 to non logo channel
    def pick_random(self, logos, array, revertOne=False):
        if len(logos) > 0:
            rand = random.choice(logos)
            array[rand[0], rand[1], rand[2]] = 1
            if revertOne:
                array[rand[0] + self.cSep, rand[1], rand[2]] -= 1
        return array


    '''
        Returns the offset of the layout for the grid calculation.
        This is done because some websites fix the logo to the far left (or right)
        and some are within a container, that usually stays fixed when increasing the resolution.
        This is done in Bootstrap for example.    
    '''
def normalize_layout(logos, screen_res = 1920):
    # Get border of container on x axis for the left side
    if screen_res > container_size:
        min_left = (1920/2) - (container_size/2)
        # Get border of container on x axis for the right side
        max_right = (1920/2) + (container_size/2)
    else:
        min_left = 0
        max_right = screen_res
    for j, label in enumerate(logos):
        values = label["value"]
        pmin = int(values["x"] * 1920)
        pmax = int(((values["x"] + values["width"]) / 100) * 1920)
        if pmin < min_left:
            # Add some margin to left (5px)
            min_left = max(0, pmin - 5)
        if pmax > max_right:
            # Add some margin to right (5px)
            max_right = min(1920, pmax + 5)
    return max(min(min_left, 1920 - max_right)/1920, 0) * 100

'''
    Calculates the binning:
        INPUT: grid resolution, normalization offset, 
'''
def calculate_bin(grid, norm_offset, values):
    # Calculate
    xdiv = (100 - 2 * norm_offset) / grid
    ydiv = 100 / grid
    return numpy.array([(values["y"] + (values["height"] / 2)) // ydiv,
                        max(0, (values["x"] - norm_offset + (values["width"] / 2))) // xdiv], dtype=int)


# Index of logo, logo or not logo channel (0 = logo, 1 = not logo)
def channel_of_logo(cSep, size_channels, logo_size, tof=1):
    add = tof * cSep
    for k, i in enumerate(size_channels):
        if logo_size < i:
            return k + add
    return len(size_channels) - 1 + add

'''
    Computes the normalization offset given a resolution. This increases accuracy due to layout shifts caused 
    by frameworks and different design principles.
        - INPUT: bounding boxes, resolution of image
        - RETURN: offset
'''
def direct_normalize_layout(logos, screen_res = 1920):
    # Get border of container on x axis for the left side
    if screen_res > container_size:
        min_left = (1920/2) - (container_size/2)
        # Get border of container on x axis for the right side
        max_right = (1920/2) + (container_size/2)
        for j, label in enumerate(logos):
            values = label
            pmin = int(values["x"] * 1920)
            pmax = int(((values["x"] + values["width"]) / 100) * 1920)
            if pmin < min_left:
                # Add some margin to left (5px)
                min_left = max(0, pmin - 5)
            if pmax > max_right:
                # Add some margin to right (5px)
                max_right = min(1920, pmax + 5)
    else:
        min_left = 0
        max_right = screen_res
    return max(min(min_left, 1920 - max_right)/1920, 0) * 100