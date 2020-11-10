# Things don't need anymore but didn't want to delete

def process_nrrds():
    segments = os.listdir(PREFIX_DIR + LM_DIR)
    names = [i.split('.')[0] for i in segments]
    segs = PREFIX_DIR + LM_DIR
    errored = set()
    for name in names:
        curr_file = segs + name + '.nrrd'
        try:
            data, _ = nrrd.read(curr_file, index_order='C')
            if data.shape == (1, 1, 1):
                errored.add(name)
            else:
                if data.shape[0] < 4:
                    print(f'{name}: {data.shape}')
                    print_nrrd_map(data, name)
        except FileNotFoundError:
            print(f'Error: could not find {curr_file}')
            errored.add(name)
    print(len(set(names).difference(errored)))

def print_image_mask(name):
    try:
        orig = Image.open(PREFIX_DIR + IMAGES_DIR + name + '.jpg')
    except FileNotFoundError:
        orig = Image.open(PREFIX_DIR + IMAGES_DIR + name + '.JPG')
    orig = orig.convert('L')
    orig = np.array(orig).astype(np.int32)
    out = np.full(orig.shape, 255).astype(np.int32)
    for i in range(orig.shape[0]):
        for j in range(orig.shape[1]):
            d = orig[i][j]
            if d < 240 and d > 45:
                out[i][j] = 0
    img = Image.fromarray(out.astype(np.uint8))
    img.save(PREFIX_DIR + 'auto_masks/' + name + '.png')

def print_nrrd_map(data, name):
    out = np.full((data.shape[1], data.shape[2]), 255).astype(np.uint8)
    for i in range(data.shape[1]):
        for j in range(data.shape[2]):
            if data[0][i][j] != 0:
                out[i][j] = 0
    img = Image.fromarray(out)
    img.save(PREFIX_DIR + 'maps_lm/' + name + '.png')

def print_image_outline(name):
    try:
        orig = Image.open(PREFIX_DIR + IMAGES_DIR + name + '.jpg')
    except FileNotFoundError:
        orig = Image.open(PREFIX_DIR + IMAGES_DIR + name + '.JPG')
    orig = orig.convert('L')
    orig = np.array(orig).astype(np.int32)
    for size in range(5, 200, 5):
        print(size)
        out = np.full(orig.shape, 255).astype(np.int32)
        for i in range(orig.shape[0]):
            for j in range(orig.shape[1]):
                d = orig[i][j]
                for k in range(-1, 2):
                    for w in range(-1, 2):
                        try:
                            if d-orig[i+k][j+w] > size:
                                out[i][j] = 0
                        except IndexError:
                            # Just means we are at an edge
                            pass
        img = Image.fromarray(out.astype(np.uint8))
        img.save(PREFIX_DIR + 'outlines/' + name + f'_{size}.png')

def overlay_mask_on_image(data, name):
    try:
        orig = Image.open(PREFIX_DIR + IMAGES_DIR + name + '.jpg')
    except FileNotFoundError:
        orig = Image.open(PREFIX_DIR + IMAGES_DIR + name + '.JPG')
    out = np.array(orig)
    for w in range(data.shape[0]):
        for i in range(data.shape[1]):
            for j in range(data.shape[2]):
                if data[w][i][j][0] > 0:
                    init = out[i][j]
                    out[i][j] = [max(x-100, 0) for x in init]
    img = Image.fromarray(out)
    img.save(PREFIX_DIR + 'overlays/' + name + '.png')
