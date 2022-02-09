import json
import os
import pandas as pd
import yaml

with open('config/mask_rcnn_R_50_FPN_3x.yaml', 'r') as f:
    iters = yaml.load(f, Loader=yaml.FullLoader)["SOLVER"]["MAX_ITER"]

with open('config/enhance.json', 'r') as f:
    enhance = json.load(f)

ENHANCE = bool(enhance['ENHANCE'])

fname = 'metadata.json'
if ENHANCE:
    fname = 'enhanced_' + fname
else:
    fname = 'non_enhanced_' + fname

with open(fname, 'r') as f:
    metadata = json.load(f)

missing_fish_count, missing_scale_count, multiple_fish_count, missing_ruler_count, missing_eye_count = 0, 0, 0, 0, 0
metadata_length = len(list(metadata.keys()))
inhs_count, uwzm_count, errored = 0, 0, 0

faulty_images = {}
for key in metadata:
    if 'errored' in metadata[key] and metadata[key]['errored']:
        errored += 1
        print(key)
        continue
    missing_fish_change, missing_scale_change, multiple_fish_change, missing_ruler_change, missing_eye_change = \
        missing_fish_count, missing_scale_count, multiple_fish_count, missing_ruler_count, missing_eye_count
    missing_ruler_count += int('has_ruler' not in metadata[key] or not metadata[key]['has_ruler'])
    # multiple_fish_count += int(metadata[key]['has_fish']
    #                           and metadata[key]['fish_count'] > 1)
    missing_fish_count += int('has_fish' not in metadata[key] or not metadata[key]['has_fish'])
    missing_scale_count += int('scale' not in metadata[key])
    missing_eye_count += int(sum(['has_eye' not in fish or not fish['has_eye'] for fish in metadata[key]['fish']]) ==
                             len(metadata[key]['fish'])) if 'fish' in metadata[key] else 1
    missing_fish_change -= missing_fish_count
    missing_scale_change -= missing_scale_count
    # multiple_fish_change -= multiple_fish_count
    missing_ruler_change -= missing_ruler_count
    missing_eye_change -= missing_eye_count
    if missing_fish_change or missing_scale_change or missing_ruler_change or missing_eye_change:
        if 'inhs' in key.lower():
            inhs_count += 1
        elif 'uwzm' in key.lower():
            uwzm_count += 1
        faulty_images[key] = {
            "missing_fish": bool(missing_fish_change),
            "missing_scale": bool(missing_scale_change),
            # "multiple_fish": bool(multiple_fish_change),
            "missing_ruler": bool(missing_ruler_change),
            "missing_eye": bool(missing_eye_change)
        }

efname = 'error.json'
if ENHANCE:
    efname = 'enhanced_' + efname
else:
    efname = 'non_enhanced_' + efname

with open(efname, 'w+') as f:
    json.dump(faulty_images, f, indent=4)

bad_length = len(list(faulty_images.keys()))


def compare():
    copy_efname = 'error_copy.json'
    if ENHANCE:
        copy_efname = 'enhanced_' + copy_efname
    else:
        copy_efname = 'non_enhanced_' + copy_efname

    with open(copy_efname, 'r') as f:
        old_data = json.load(f)
    print(len(old_data))
    print(len([k for k in old_data if k not in faulty_images and old_data[k]['missing_eye']]))


def main():
    csv_columns = ['Iters', 'Total', 'No Fish',
                   'No Ruler', 'No Eye', 'Multiple Fish', 'No Scale']
    data = {
        'Iters': iters,
        'Total': bad_length,
        'No Fish': missing_fish_count,
        'No Ruler': missing_ruler_count,
        'No Eye': missing_eye_count,
        'Multiple Fish': 0,  # multiple_fish_count,
        'No Scale': missing_scale_count,
    }
    print(data)
    df_data = pd.DataFrame(data=data, index=[0])
    dfname = 'error.csv'
    if ENHANCE:
        dfname = 'enhanced_' + dfname
    else:
        dfname = 'non_enhanced_' + dfname

    df_file = pd.read_csv(dfname, header=0) if os.path.isfile(
        dfname) else pd.DataFrame(columns=csv_columns)
    if not (df_file == df_data.values).all(1).any():
        df_file = pd.concat([df_file, df_data])
    df_file.to_csv(dfname, index=False, float_format="%.2f")
    total_count = missing_fish_count + missing_scale_count + multiple_fish_count + missing_ruler_count + \
                  missing_eye_count
    print(f"Erroneous Image Count: {bad_length}")
    print(f'Actual Error Count: {total_count}')
    print(f"INHS Image Count: {inhs_count}")
    print(f"UWZM Image Count: {uwzm_count}")
    print(f"Total Image Count: {metadata_length}")
    print(f'Errored: {errored}')


if __name__ == "__main__":
    main()
