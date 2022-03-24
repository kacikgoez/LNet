from tqdm import tqdm
import os
import numpy as np
from PIL import Image

'''
    Identifies images with transparent backgrounds and kindly
        asks you to replace them with a non-transparent image
'''


def angle_between_vectors(vec1, vec2):
    unit_vector_1 = vec1 / np.linalg.norm(vec1)
    unit_vector_2 = vec2 / np.linalg.norm(vec2)
    dot_product = np.dot(unit_vector_1, unit_vector_2)
    angle = np.arccos(dot_product)
    return np.rad2deg(angle)

def has_transparency(img):
    if img.mode == "P":
        transparent = img.info.get("transparency", -1)
        for _, index in img.getcolors():
            if index == transparent:
                return True
    elif img.mode == "RGBA":
        extrema = img.getextrema()
        if extrema[3][0] < 255:
            return True
    return False

if __name__ == "__main__":
    # Path to scan
    targetlist_path = r"../src/siamese_pedia/expand_targetlist/"

    for target in tqdm(os.listdir(targetlist_path)):
        if target.startswith('.'):  # skip hidden files
            continue
        for logo_path in os.listdir(os.path.join(targetlist_path, target)):
            if logo_path.endswith('.png') or logo_path.endswith('.jpeg') or logo_path.endswith(
                    '.jpg') or logo_path.endswith('.PNG') \
                    or logo_path.endswith('.JPG') or logo_path.endswith('.JPEG'):
                if logo_path.startswith('loginpage') or logo_path.startswith('homepage'):  # skip homepage/loginpage
                    continue
                # Check if file is already in existing stored db
                completePath = os.path.join(r'../src/siamese_pedia/expand_targetlist/', target, logo_path)
                img = Image.open(completePath)
                if has_transparency(img):
                    print(completePath)
                    all_colors = np.unique(np.asarray(img).reshape((-1, 4)), axis=0)
                    '''for i in all_colors:
                        if angle_between_vectors(i, [255, 255, 255, 255]) < 30:
                            print("Test")
                            break'''