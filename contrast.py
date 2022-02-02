import numpy as np
import json
import cv2
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import PercentFormatter

prefix = '/home/HDD/bgnn_data/validation_images/'
results = json.load(open('../metadata_enhance.json'))
df = pd.read_csv('/home/HDD/bgnn_data/image_quality_metadata_20210208.csv')
values = []
#d = []
#n = []
#b = []

for file in results.keys():
    try:
        curr = results[file]
        back = curr['fish'][0]['background']['mean']
        fore = curr['fish'][0]['foreground']['mean']
        diff = back - fore
        if diff > 0:
            values.append((file, diff))
    except:
        pass

#values.sort(key=lambda x: x[1], reverse=True)
# print(values)
# exit(0)

print(np.mean([x[1] for x in values]))
print(np.std([x[1] for x in values]))
exit(0)
plt.xlabel('Foreground Background Intensity Difference')
plt.ylabel('Count')
plt.hist([x[1] for x in values], bins=20)
plt.show()
