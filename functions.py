from PIL import Image
import os
import glob
import sys
sys.path.insert(0,'/content/drive/MyDrive/job_tasks/liveeo')

def expand2square(pil_img, background_color):
    """A function to resize the images and labels to 1024x1024"""
    width, height = pil_img.size
    if width == 1023 and height == 1023:
        result = Image.new(pil_img.mode, (width + 1, width + 1), background_color)
        result.paste(pil_img, (0, 1 // 2))
        return result
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result
    
def path_dir(plist, pmask):
    """A function that creates the list of sorted paths to the images and labels (masks)"""
    path = ''
    image_path = os.path.join(path, plist)
    mask_path = os.path.join(path, pmask)
    os.chdir(image_path)
    image_sorted = sorted(glob.glob('*.tif'))
    os.chdir('../')
    os.chdir(mask_path)
    mask_sorted = sorted(glob.glob('*.tif'))
    image_list = [image_path+i for i in image_sorted]
    mask_list = [mask_path+i for i in mask_sorted]
    os.chdir('../')
    return image_list, mask_list