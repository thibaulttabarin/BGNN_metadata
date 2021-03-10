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
no_eye = 0
no_fish = 0
errored = 0
for i in results.keys():
    counter += 1
    me = results[i]
    #print(me)
    if not 'errored' in me.keys():
        yasin = df[df.image_name == i].iloc[0]
        if me['has_fish']:
            if me['fish'][0]['has_eye']:
                val = int(me['fish'][0]['clock_value'])
                print(yasin)
                exit(0)
                right += val >= round(yasin.specimen_angled - 1) and\
                        val <= round(yasin.specimen_angled + 1)
                print(f"{i}: {me['fish'][0]['clock_value']}, {yasin.specimen_angled}")
            else:
                no_eye += 1
                print(f'{i}: No eye')
        else:
            no_fish += 1
            print(f'{i}: No fish')
        #print(me['fish']==)
    else:
        errored += 1
        print(f'{i}: Errored out')
print(f'\nRight: {right}')
print(f'Errored: {errored}')
print(f'No eye: {no_eye}')
print(f'No fish: {no_fish}')
print(f'Total: {counter}')
print(f'Percent right: {right / counter}')
print(f'Percent right that didn\'t error: {right / (counter - no_eye - no_fish - errored)}')
