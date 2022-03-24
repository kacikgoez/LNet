# Global configuration
import sys
sys.path.append("../")
sys.path.append("lnet")
from src.siamese import *
from src.detectron2_pedia.inference import *
from layoutnet.main import WebLayoutNet5, WebLayoutNet7
import torch
import keras_ocr


# Define keras pipeline
keras_pipeline = keras_ocr.pipeline.Pipeline()

# element recognition model -- logo only
cfg_path = '../src/detectron2_pedia/configs/faster_rcnn.yaml'
# navbar, info, button, popup recongnition
cfg_path2 = './navbar/configs/faster_rcnn.yaml'
# Weights to logo network
weights_path = '../src/detectron2_pedia/output/rcnn_2/rcnn_bet365.pth'
# Weights advanced network
weights_path2 = './layoutnet/model_0020499.pth'

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Decreased from 0.01 to 0.05
#lnet_ele_model = config_rcnn(cfg_path, device, weights_path, conf_threshold=0.01)

#Lnet config
ele_model = config_rcnn(cfg_path, device, weights_path, conf_threshold=0.05)

# Further
nav_model = config_rcnn(cfg_path2, device, weights_path2, conf_threshold=0.05)

layout_model = WebLayoutNet5().cuda() if device == 'cuda' else WebLayoutNet5()
layout_model.load_state_dict(torch.load("./layoutnet/model-final-5x5.pth", map_location=device))
layout_model.to(device)
# Evaluation mode to avoid batch norm issue
layout_model.eval()

layout_model_7 = WebLayoutNet7().cuda() if device == 'cuda' else WebLayoutNet7()
layout_model_7.load_state_dict(torch.load("./layoutnet/model-final-7x7.pth", map_location=device))
layout_model_7.to(device)
# Evaluation mode to avoid batch norm issue
layout_model_7.eval()

# siamese model
print('Load protected logo list')
pedia_model, logo_feat_list, file_name_list, word_list = phishpedia_config(num_classes=277,
                                                weights_path='../src/siamese_pedia/resnetv2_rgb_new.pth.tar',
                                                targetlist_path=r"../src/siamese_pedia/expand_targetlist",
                                                keras_pipeline=keras_pipeline)
print(word_list)

print('Finish loading protected logo list')
print(logo_feat_list.shape)

# Before it was tested on 0.91
lnet_siamese_ts = 0.75
lnet_upper_ts = 0.87
siamese_ts = 0.83

# brand-domain dictionary
domain_map_path = '../src/siamese_pedia/domain_map.pkl'

