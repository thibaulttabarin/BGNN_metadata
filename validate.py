import json
import pandas as pd
import numpy as np
import os

df = pd.read_csv('/home/HDD/bgnn_data/image_quality_metadata_20210208.csv')
results = json.load(open('metadata.json'))

#one=set(os.listdir('/home/HDD/bgnn_data/full_imgs/'))
#two=set(df.image_name)
#inter=list(one&two)

#np.random.seed(1)
#values = np.unique(np.random.randint(len(inter), size=100))
#for i in values:
    #print(inter[i])
#exit(0)

counter = 0
right = 0
errored = 0
for i in results.keys():
    counter += 1
    me = results[i]
    #print(me)
    if not 'errored' in me.keys():
        yasin = df[df.image_name == i].iloc[0]
        if me['has_fish']:
            if me['fish'][0]['has_eye']:
                right += me['fish'][0]['clock_value'] == yasin.specimen_angled
                print(f"{i}: {me['fish'][0]['clock_value']}, {yasin.specimen_angled}")
            else:
                errored += 1
        #print(me['fish']==)
    else:
        errored += 1
print(f'Right: {right}')
print(f'Failed: {errored}')
print(f'Total: {counter}')
print(f'Percent Right: {right / counter}')
