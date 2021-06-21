import numpy as np
import json
import cv2
import pandas as pd

prefix = '/home/HDD/bgnn_data/validation_images/'
results = json.load(open('../metadata.json'))
df = pd.read_csv('/home/HDD/bgnn_data/image_quality_metadata_20210208.csv')
#d = []
#n = []
#b = []
#mm = []
right = 0
wrong = 0
mm = []
f = open('brightness.csv', 'w')

for file in results.keys():
    try:
        #file_path = prefix + file
        brightness = df[df.image_name == file].iloc[0].brightness
        if brightness == 'dark':
            bright = 0
        elif brightness == 'normal':
            bright = 1
        elif brightness == 'bright':
            bright = 2
        else:
            bright = None
        if bright is not None:
            #bbox = results[file]['fish'][0]['bbox']
            #im_gray = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE).reshape(-1)
            curr = results[file]
            back = curr['fish'][0]['background']['mean']
            fore = curr['fish'][0]['foreground']['mean']
            m = back - fore
            if m > 0:
                m = fore
                #mm.append(m)
                f.write(f'{bright},{m}\n')
                """
                if m < 78.5:
                    guess = 0
                elif m >= 78.5 and m < 100.5:
                    guess = 1
                else:
                    guess = 2
                mm.append((guess, bright))
                """
                #if brightness == 'dark':
                    #d.append(m)
                #elif brightness == 'normal':
                    #n.append(m)
                #elif brightness == 'bright':
                    #b.append(m)
                #else:
    except:
        pass

f.close()
"""
mat = np.zeros((3, 3))
for i in mm:
    mat[i[0],i[1]] += 1
print(mat)
"""

#print(f'Right: {right}')
#print(f'Wrong: {wrong}')
#print(f'Total: {right + wrong}')
#print(f'overall: {np.mean(mm)} | {np.std(mm)} | {len(mm)}')
#print(f'dark: {np.mean(d)} | {np.std(d)} | {len(d)}')
#print(f'normal: {np.mean(n)} | {np.std(n)} | {len(n)}')
#print(f'bright: {np.mean(b)} | {np.std(b)} | {len(b)}')
