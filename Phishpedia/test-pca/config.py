# Global configuration
import sys
sys.path.append('../')
from src.siameseQuick import *

'''
    Loading both classifiers for PCA and original Phishpedia
'''

pcaActive = True

# siamese model
print('Load protected logo list')
pca_model, logo_feat_list, file_name_list, pca = phishpedia_config(num_classes=277,
                                                weights_path='../src/siamese_pedia/resnetv2_rgb_new.pth.tar',
                                                targetlist_path='../src/siamese_pedia/expand_targetlist/',
                                                pcaActive=pcaActive)

pedia_model, logo_feat_list2, file_name_list2, _ = phishpedia_config(num_classes=277,
                                                weights_path='../src/siamese_pedia/resnetv2_rgb_new.pth.tar',
                                                targetlist_path='../src/siamese_pedia/expand_targetlist/',
                                                pcaActive=False)
if pcaActive is False:
    pca = None

siamese_ts = 0.87 # FIXME: threshold is 0.87 in phish-discovery?

# brand-domain dictionary
domain_map_path = '../src/siamese_pedia/domain_map.pkl'


