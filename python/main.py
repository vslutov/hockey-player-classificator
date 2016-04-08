#!/usr/bin/env python3

import sys
import os.path
import itertools

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

import skimage.filters
import skimage.segmentation
import skimage.measure
import skimage.color

def get_filepath(source_dir, filename):
    return os.path.abspath(os.path.join(os.path.expanduser(source_dir), filename))

def markup_dir(hockey_dir):
    image_dir = get_filepath(hockey_dir, 'images')
    sample_num = 0
    sample_dir = get_filepath(hockey_dir, 'sample')
    os.makedirs(sample_dir, exist_ok=True)
    
    for i in itertools.count(0, 10):
        try:
            frame = skimage.io.imread(get_filepath(image_dir, 'frame_{i}.png'.format(i=i)))
            mask = skimage.io.imread(get_filepath(image_dir, 'mask_{i}.png'.format(i=i)))
        
        except FileNotFoundError:
            break
       
        # apply threshold
        thresh = skimage.filter.threshold_otsu(mask)
        bw = skimage.morphology.closing(mask > thresh, skimage.morphology.square(3))

        # remove artifacts connected to image border
        cleared = bw.copy()
        skimage.segmentation.clear_border(cleared)
                
        # label image regions
        label_image = skimage.measure.label(cleared)        
        
        for region in skimage.measure.regionprops(label_image):
            if region.area < 250:
                continue
    
            # draw rectangle around segmented coins
            minr, minc, maxr, maxc = region.bbox
            
            sample = frame[minr:maxr, minc:maxc]
            sample_mask = cleared[minr:maxr, minc:maxc]
            
            skimage.io.imsave(get_filepath(sample_dir, 'sample_{sample_num}.png'.format(sample_num=sample_num)), sample)
            skimage.io.imsave(get_filepath(sample_dir, 'sample_mask_{sample_num}.png'.format(sample_num=sample_num)), sample_mask)
            
            sample_num += 1            
        
        print(i)

def main():
    if len(sys.argv) != 2:
        print("usage: {command} hockey_dir".format(command=sys.argv[0]))
        sys.exit(1)
    
    hockey_dir = sys.argv[1]
    
    markup_dir(hockey_dir)

if __name__ == '__main__':
    main()