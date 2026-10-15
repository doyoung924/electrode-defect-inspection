from pillow_heif import register_heif_opener; register_heif_opener()
from PIL import Image
import numpy as np
UP='/mnt/user-data/uploads/'
B1={'N03':'IMG_7273','N04':'IMG_7278','N05':'IMG_7284','N06':'IMG_7286','N07':'IMG_7298','N08':'IMG_7299',
    'N09':'IMG_7301','N10':'IMG_7302','N11':'IMG_7303','N12':'IMG_7305','N13':'IMG_7328','N14':'IMG_7329',
    'N15':'IMG_7330','N16':'IMG_7332','N17':'IMG_7335','N18':'IMG_7334'}
B2={'N19':'IMG_7339','N20':'IMG_7340','N21':'IMG_7342','N22':'IMG_7341','N23':'IMG_7343','N24':'IMG_7345',
    'N25':'IMG_7438','N26':'IMG_7439','N27':'IMG_7440','N28':'IMG_7441','N29':'IMG_7442','N30':'IMG_7443',
    'N31':'IMG_7451','N32':'IMG_7452','N33':'IMG_7453','N34':'IMG_7455','N35':'IMG_7456','N36':'IMG_7460'}
P={'P01':'IMG_7467','P02':'IMG_7473','P03':'IMG_7475','P04':'IMG_7477','P05':'IMG_7480','P06':'IMG_7485','P07':'IMG_7504','P08':'IMG_7507','P09':'IMG_7509','P11':'IMG_7513','P12':'IMG_7515'}
ALL={**B1,**B2,**P}
GRADE={'N20':'A','N21':'A','N22':'A','N24':'A','N27':'A','N28':'A','N30':'A','N32':'A','N35':'A','N19':'B','N23':'B','N25':'B','N26':'B','N31':'B','N33':'B','N34':'B','N36':'B'}
KIND={'N08':'이물','N12':'이물','N15':'줄무늬','N17':'줄무늬','N29':'줄무늬'}
def load(fid):
    im=Image.open(UP+fid+'.HEIC'); return im
