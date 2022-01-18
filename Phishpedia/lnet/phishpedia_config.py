# Global configuration
import sys
sys.path.append("../")
sys.path.append("lnet")
from src.siamese import *
from src.detectron2_pedia.inference import *
from layoutnet.main import WebLayoutNet
import torch

# element recognition model -- logo only
cfg_path = '../src/detectron2_pedia/configs/faster_rcnn.yaml'
weights_path = '../src/detectron2_pedia/output/rcnn_2/rcnn_bet365.pth'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Increased conf threshold to 0.09 from 0.05
ele_model = config_rcnn(cfg_path, device, weights_path, conf_threshold=0.05)

layout_model = WebLayoutNet().cuda() if device == 'cuda' else WebLayoutNet()
layout_model.load_state_dict(torch.load("./layoutnet/models/final-3d-7c-3s.pth", map_location=device))
layout_model.to(device)
# Evaluation mode to avoid batch norm issue
layout_model.eval()


# siamese model
print('Load protected logo list')
pedia_model, logo_feat_list, file_name_list = phishpedia_config(num_classes=277,
                                                weights_path='../src/siamese_pedia/resnetv2_rgb_new.pth.tar',
                                                targetlist_path='../src/siamese_pedia/expand_targetlist/')


print('Finish loading protected logo list')
print(logo_feat_list.shape)

# Before it was tested on 0.91
siamese_ts = 0.87 # FIXME: threshold is 0.87 in phish-discovery?

# brand-domain dictionary
domain_map_path = '../src/siamese_pedia/domain_map.pkl'

