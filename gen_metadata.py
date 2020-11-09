#!/bin/python3

import sys
import math
from PIL import ImageStat, Image

NORMAL = 0.8
BRIGHT = 0.85

def raw_brightness(im):
    stat = ImageStat.Stat(im)
    if im.mode != 'L':
        r,g,b = stat.mean
    else:
        r = g = b = stat.mean[0]
    # https://stackoverflow.com/questions/3490727/what-are-some-methods-to-analyze-image-brightness-using-python
    # https://www.nbdtech.com/Blog/archive/2008/04/27/Calculating-the-Perceived-Brightness-of-a-Color.aspx
    brightness = math.sqrt(0.241*(r**2)+.691*(g**2)+.068*(b**2))
    return brightness / 255.0

def brightness(im):
    brightness = raw_brightness(im)
    if brightness > BRIGHT:
        result = 'bright'
    elif brightness > NORMAL:
        result = 'normal'
    else:
        result = 'dark'
    return result

def is_radiograph(im):
    # Really detecting if there's (much) color
    if im.mode == 'L':
        return True
    width, height = im.size
    diff = 0
    for j in range(height):
        for i in range(width):
            r, g, b = im.getpixel((i, j))
            diff += abs(r - g) + abs(r - b) + abs(g - b)

    avg = diff / (height * width) / 765.0
    return avg > 0.002

if __name__ == '__main__':
    im = Image.open(sys.argv[1])
    #print(f'{sys.argv[1]} is {brightness(im)}')
    print(f'{sys.argv[1]} is {is_radiograph(im)}')
