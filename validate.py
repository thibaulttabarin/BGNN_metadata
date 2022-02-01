import json
import pandas as pd
import numpy as np
import os

df = pd.read_csv('/usr/local/bgnn/image_quality_metadata_20210208.csv')
results = json.load(open('enhanced_error.json'))
inhs_results = {}
uwzm_results = {}
for key in results:
    if "INHS" in key:
        inhs_results[key] = results[key]
    elif "UWZM" in key:
        uwzm_results[key] = results[key]
    else:
        print("sum ting wong")

results = inhs_results

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
wrong_wrong = 0
side_wrong = 0
length_calced = 0
bbox = 0
for i in results.keys():
    counter += 1
    me = results[i]
    #print(me)
    if not 'errored' in me.keys():
        all_yasin = df[df.image_name == i]
        all_yasin = all_yasin[~all_yasin['specimen_angled'].isnull()]
        #all_yasin = all_yasin[not all_yasin.specimen_angled.isnull()]
        #print(all_yasin.specimen_angled)
        #yasin = df[df.image_name == i].iloc[0]
        angle = np.mean(all_yasin['specimen_angled'])
        #print(angle)
        angle = round(angle)
        if me['has_fish']:
            if 'bbox' in me['fish'][0].keys():
                bbox += 1
            if 'length' in me['fish'][0].keys():
                length_calced += 1
            if me['fish'][0]['has_eye']:
                val = int(me['fish'][0]['clock_value'])
                #print(yasin)
                #exit(0)
                check = (angle - 1) <= val <= (angle + 1)
                right += check
                wrong_wrong += 1 if not check else 0
                #right += val >= round(yasin.specimen_angled - 1) and\
                #        val <= round(yasin.specimen_angled + 1)
                #if not (val >= (angle - 1) and val <= (angle + 1)):
                    #print(f"{i}: {me['fish'][0]['clock_value']}, {yasin.specimen_angled}")
                    #print(f"{i}: {me['fish'][0]['clock_value']}, {angle}")
                mes = me['fish'][0]['side']
                yasins = all_yasin.iloc[0].specimen_viewing
                if yasins.find(mes) < 0:
                    side_wrong += 1
                    print(f'{i}: Side wrong | {mes} | {yasins}')
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
print(f'Clock wrong: {wrong_wrong}')
print(f'Total: {counter}')
print(f'Percent right: {right / counter}')
print(f'Percent right that didn\'t error: {right / (counter - no_eye - no_fish - errored)}')
print(f'\nSide wrong: {side_wrong}')
print(f'Length calculated: {length_calced}')
print(f'Bbox calculated: {bbox}')
