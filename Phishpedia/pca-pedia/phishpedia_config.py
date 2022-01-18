# Global configuration
import sys
sys.path.append('../')

from src.siameseQuick import *
from src.detectron2_pedia.inference import *

# element recognition model -- logo only
cfg_path = '../src/detectron2_pedia/configs/faster_rcnn.yaml'
weights_path = '../src/detectron2_pedia/output/rcnn_2/rcnn_bet365.pth'


device = 'cuda' if torch.cuda.is_available() else 'cpu'
ele_model = config_rcnn(cfg_path, device, weights_path, conf_threshold=0.05)


pcaActive = False

# siamese model
print('Load protected logo list')
pedia_model, logo_feat_list, file_name_list, pca = phishpedia_config(num_classes=277,
                                                weights_path='../src/siamese_pedia/resnetv2_rgb_new.pth.tar',
                                                targetlist_path='../src/siamese_pedia/expand_targetlist/',
                                                pcaActive=pcaActive)

if pcaActive is False:
    pca = None

print('Finish loading protected logo list')
print(logo_feat_list.shape)

siamese_ts = 0.87 # FIXME: threshold is 0.87 in phish-discovery?

# brand-domain dictionary
domain_map_path = '../src/siamese_pedia/domain_map.pkl'


