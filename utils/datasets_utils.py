from PIL import ImageFont, ImageDraw, Image
import numpy as np
from torchvision import transforms as T
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import utils.gen_synthetic_segments as gss



GRAY_TRANSFORM = T.Compose([
    T.ToTensor(),
    T.Grayscale(num_output_channels=3),
    T.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD),
])


INV_NORMALIZE = T.Normalize(
   mean= [-m/s for m, s in zip(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)],
   std= [1/s for s in IMAGENET_DEFAULT_STD]
)






class MedianPad:

    def __init__(self, override=None):

        self.override = override

    def __call__(self, image):

        ##Convert to RGB 
        image = image.convert("RGB") if isinstance(image, Image.Image) else image
        image = Image.fromarray(image) if isinstance(image, np.ndarray) else image
        max_side = max(image.size)
        pad_x, pad_y = [max_side - s for s in image.size]
        # padding = (0, 0, pad_x, pad_y)
        padding = (round((10+pad_x)/2), round((5+pad_y)/2), round((10+pad_x)/2), round((5+pad_y)/2)) ##Added some extra to avoid info on the long edge

        imgarray = np.array(image)
        h, w , c= imgarray.shape
        rightb, leftb = imgarray[:,w-1,:], imgarray[:,0,:]
        topb, bottomb = imgarray[0,:,:], imgarray[h-1,:,:]
        bordervals = np.concatenate([rightb, leftb, topb, bottomb], axis=0)
        medval = tuple([int(v) for v in np.median(bordervals, axis=0)])

        return T.Pad(padding, fill=medval if self.override is None else self.override)(image)






BASE_TRANSFORM = T.Compose([
        MedianPad(override=(255,255,255)),
        T.ToTensor(),
        T.Resize((224, 224)),
        T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
])

NO_PAD_NO_RESIZE_TRANSFORM = T.Compose([
    ##To rgb
        T.Lambda(lambda x: x.convert("RGB") if isinstance(x, Image.Image) else x),
        T.ToTensor(),
        T.Resize(224,max_size=225),
        T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
])

LIGHT_AUG_BASE= T.Compose([T.Normalize(
   mean= [-m/s for m, s in zip(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)],
   std= [1/s for s in IMAGENET_DEFAULT_STD]),gss.LIGHT_AUG, BASE_TRANSFORM])

EMPTY_TRANSFORM = T.Compose([
        T.ToTensor()
])


PAD_RESIZE_TRANSFORM = T.Compose([
        MedianPad(override=(255,255,255)),
        T.ToTensor(),
        T.Resize((224, 224))
])


AUG_NORMALIZE = T.Compose([gss.get_render_transform(), T.ToTensor(),
 T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)])


RANDOM_TRANS_PAD_RESIZE= T.Compose([gss.get_render_transform(), PAD_RESIZE_TRANSFORM])

def create_random_doc_transform(size=224):
    return T.Compose([  gss.get_render_transform(), BASE_TRANSFORM])


def create_clip_random_doc_transform(size=224):
    return T.Compose([  gss.get_render_transform(), CLIP_BASE_TRANSFORM])


def create_random_no_aug_doc_transform(size=224):
    return T.Compose([  gss.get_render_transform(), BASE_TRANSFORM])
    
def create_random_doc_transform_no_resize():
    return T.Compose([  gss.get_render_transform(), NO_PAD_NO_RESIZE_TRANSFORM])

CLIP_BASE_TRANSFORM = T.Compose([  
         MedianPad(override=(255,255,255)),
        T.ToTensor(),
        T.Resize((224, 224)),
        T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711),)
    ])

CLIP_BASE_TRANSFORM_RANDOM = T.Compose([  
        ###REsize shortest edge
        T.Lambda(lambda x: x.convert("RGB") if isinstance(x, Image.Image) else x),
        T.ToTensor(),
        ##RAndom crop
        T.RandomCrop((224, 224),pad_if_needed=True),
        T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711),)
    ])

CLIP_BASE_TRANSFORM_CENTER= T.Compose([  
        T.Lambda(lambda x: x.convert("RGB") if isinstance(x, Image.Image) else x),
        T.ToTensor(),
        ###REsize shortest edge
        T.Resize((224)),
        ##CEnter crop
        T.CenterCrop((224, 224)),
        T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711),)
    ])
