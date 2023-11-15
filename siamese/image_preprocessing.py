# import libraries
import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from pydicom import dcmread

# Utils
class config():
    sizes = [224, 512]
    status = os.listdir("data")
    vmin, vmax = 0, 2.5

def normalization(img):
    img = img - img.min()
    img = img.max()
    return img

def color_balanced(img, vmin, vmax):
    new_img = (img - vmin) / (vmax - vmin)
    return(new_img)

def check_log_transform():

    done = [0, 0, 0]

    for i, cate in enumerate(config.status):
        patients = os.listdir(f"./data/{cate}")
        for patient in patients:
            files = os.listdir(f"./data/{cate}/{patient}")
            for file in files:
                img = dcmread(f"./data/{cate}/{patient}/{file}")
                try:
                    print(img.get(0x20500020).value)
                except AttributeError:
                    done[i] += 1

    return done

def save_to_jpg(size):
    for cate in config.status:
        patients = sorted(os.listdir(f"./data/{cate}"), key = lambda x : int(x))
        for patient in patients:
            files = os.listdir(f"./data/{cate}/{patient}")
            for file in files:
                img = dcmread(f"./data/{cate}/{patient}/{file}").pixel_array
                img = normalization(img)
                img = color_balanced(img, config.vmin, config.vmax)
                img = np.clip(img, 0, 1)
                img = cv2.resize(img, (size, size))

                file = file.replace(".dcm", ".jpg")
                filepath = f"./{size}/{cate}/{patient}"
                if os.path.isdir(filepath) == False:
                    os.makedirs(filepath)

                plt.imsave(f"{filepath}/{file}", img, cmap = "gray")

def save_to_csv():
    
    csv_file = pd.DataFrame()
    before = [] 
    after = [] 
    partial = [] 
    progressive = [] 
    stable = []

    for cate in config.status:

        patients = sorted(os.listdir(f"./data/{cate}"), key = lambda x : int(x))
        
        for patient in patients:
            if cate == "Partial response":
                partial.append(1.)
                progressive.append(0.)
                stable.append(0.)
            elif cate == "Progressive disease":
                partial.append(0.)
                progressive.append(1.)
                stable.append(0.)
            else:
                partial.append(0.)
                progressive.append(0.)
                stable.append(1.)

            files = os.listdir(f"./data/{cate}/{patient}")
            
            file_1 = files[0].replace(".dcm", ".jpg")
            file_2 = files[1].replace(".dcm", ".jpg")

            before.append(f"{cate}/{patient}/{file_1}")
            after.append(f"{cate}/{patient}/{file_2}")

    csv_file["filepath_before"] = before
    csv_file["filepath_after"] = after
    csv_file["partial_response"] = partial
    csv_file["progressive_disease"] = progressive
    csv_file["stable_disease"] = stable

    csv_file.to_csv("lung_cancer.csv", index = False)


# check if log transform has been done
check_log_transform() # [198, 200, 200]

# save images as jpg files and construct the dataframe 
def main():
    save_to_jpg(config.sizes[0])
    save_to_jpg(config.sizes[1])
    save_to_csv()

if __name__ == "__main__":
    main()





# log transform for some DICOMs
# 做log transform的話就不做normalization!

def log_transform(dcm_path):

    img = dcmread(dcm_path)

    # values
    WW = img.WindowWidth
    WC = img.WindowCenter
    BitsStored = img.BitsStored
    iMax = WC + (WW / 2)
    iMin = WC - (WW / 2)

    # transform
    img = img.pixel_array
    new_img = np.clip(img, iMin, iMax)
    new_img = - np.log((1 + new_img)/(2 ** BitsStored))

    return(new_img)

