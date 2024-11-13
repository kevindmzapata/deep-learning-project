import numpy as np
import pandas as pd
import os
import re
import cv2 as cv

def create_database(directory='16GBd_0km', n_samples=12096):
    check_if_exists = os.path.exists('images')
    if check_if_exists and len(os.listdir('images')) > 0:
        print('Images already exist in the directory')
        return
    pattern_1 = r'consY(\d+)\.(\d+)dB'
    pattern_2 = r'consY(\d+)dB'
    for dirs in os.listdir(directory)[::-1]:
        for files in os.listdir(os.path.join(directory, dirs)):
            try:
                match = re.search(pattern_1, files)
                dB = f'{match.group(1)}.{match.group(2)}'
            except:
                match = re.search(pattern_2, files)
                dB = f'{match.group(1)}'
            finally:
                dB = float(dB)
                if dirs == 'single_ch':
                    dB += 3.7
                
            path = os.path.join(directory, dirs, files)
            data = pd.read_csv(path)
            data = data.to_numpy()
            
            for i in range(0, data.shape[0], n_samples):
                data_i = data[i:i + n_samples]
                x, y = data_i[:, 0], data_i[:, 1]
                hist, _, _ = np.histogram2d(x, y, bins=32, range=[[-5, 5], [-5, 5]])
                hist = cv.normalize(hist, None, 0, 255, cv.NORM_MINMAX, cv.CV_8U)
                
                if not os.path.exists('images'):
                    os.makedirs('images')
                    
                cv.imwrite(f'images/{dirs}_{dB}dB_sample({i//n_samples}).png', hist)
                
    print('Images have been created successfully')
    return

def load_data(directory='images'):
    images = []
    labels = []
    ch_s = []
    spacings = ['single_ch', '18GHz', '17p6GHz', '17GHz', '16p5GHz', '16GHz', '15p5GHz', '15GHz']

    for files in os.listdir(directory) [::-1]:
        image = cv.imread(os.path.join(directory, files), cv.IMREAD_GRAYSCALE)
        images.append(image)
        
        for i, spacing in enumerate(spacings):
            if spacing in files:
                ch_s.append(i)
        
        match = re.search(r'(\d+\.\d+)dB', files)
        labels.append(float(match.group(1)))
        
    images = np.array(images)
    labels = np.array(labels)
    ch_s = np.array(ch_s)
    return images, labels, ch_s
