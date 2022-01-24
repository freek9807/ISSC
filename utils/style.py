import matplotlib.font_manager
from numpy import random
from os import listdir
from os.path import isfile, join

def check_is_accepted(accepted, font, type):
    okay = False
    for acc in accepted:
        if "-".join([acc,type]) in font:
            okay = True
            break
    return okay

def get_theme():
    colors = [
        ((0, 0, 0), '#FFFFFF'),
        ((255, 255, 255), '#000000'),
        ((18, 0, 222), '#FFFF00'),
        ((114, 134, 139), '#D3F6FC'),
        ((18, 0, 222), '#FFFFFF'),
        ((0, 0, 0), '#4AF626'),
        ((255, 204, 204), '#404040')
    ]
    value = random.randint(7, size=1)[0]
    return  colors[value] 

def getFonts():
    path = "fonts/"
    fonts = [path + font for font in listdir(path) if isfile(join(path, font))]
    return fonts

def getFontSize():
    return 13