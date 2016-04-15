#!/usr/bin/env python3

import sys
import os.path
import itertools
import csv

if sys.version_info[0] < 3:
    import Tkinter as Tk
else:
    import tkinter as Tk

import numpy as np

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

from skimage import filters, segmentation, measure, io, morphology, color

def get_filepath(source_dir, filename):
    return os.path.abspath(os.path.join(os.path.expanduser(source_dir), filename))

def markup(hockey_dir, ax):
    class Updater:
        value = None
    updater = Updater()

    image_dir = get_filepath(hockey_dir, 'images')
    sample_num = 0
    sample_dir = get_filepath(hockey_dir, 'sample')
    os.makedirs(sample_dir, exist_ok=True)

    gt_filepath = get_filepath(image_dir, 'gt.txt')
    start_sample = 0

    with open(gt_filepath, 'r') as gt:
        for line in csv.reader(gt):
            start_sample = max(start_sample, int(line[0]))

    with open(gt_filepath, 'a') as gt:
        for i in itertools.count(0, 40):
            try:
                frame_filename = 'frame_{i}.png'.format(i=i)
                frame = io.imread(get_filepath(image_dir, frame_filename))
                mask = io.imread(get_filepath(image_dir, 'mask_{i}.png'.format(i=i)))
                mask = color.rgb2gray(mask)

            except FileNotFoundError:
                break


            # apply threshold
            border = 0.05 * mask.shape[1]
            bw = morphology.opening(mask[:, border:-border] > 128, morphology.disk(3))
            # remove artifacts connected to image border
            segmentation.clear_border(bw)
            cleared = np.zeros(mask.shape, dtype=np.bool)
            cleared[:, border:-border] = bw

            # label image regions
            label_image = measure.label(cleared)

            for region in measure.regionprops(label_image):
                if region.area < 250:
                    continue

                if sample_num > start_sample:
                    # draw rectangle around segmented coins
                    minr, minc, maxr, maxc = region.bbox

                    sample = frame[minr:maxr, minc:maxc]
                    sample_mask = cleared[minr:maxr, minc:maxc].astype(np.uint8) * 255
                    sample_mask = np.dstack((sample_mask,) * 3)
                    io.imsave(get_filepath(sample_dir, 'sample_{sample_num}.png'.format(sample_num=sample_num)), sample)
                    io.imsave(get_filepath(sample_dir, 'sample_mask_{sample_num}.png'.format(sample_num=sample_num)), sample_mask)

                    # show_image = frame.copy()
                    # darken = np.logical_not(region.filled_image)
                    # print(show_image[darken])
                    # show_image[darken] = 0 # show_image[darken] / 2

                    rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='red', linewidth=2)

                    ax.cla()
                    ax.imshow(frame)
                    ax.add_patch(rect)

                    yield updater
                    print(sample_num, frame_filename, minr, minc, maxr, maxc, updater.value, sep=',', file=gt)

                sample_num += 1
                print(sample_num)

def user_interface(hockey_dir, interface_generator):

    root, canvas, toolbar = None, None, None

    f = Figure(figsize=(5, 4), dpi=100)
    ax = f.add_subplot(111)
    interface_generator = interface_generator(hockey_dir, ax)
    updater = next(interface_generator)

    def update(value):
        nonlocal updater
        updater.value = value
        _quit()

        try:
            updater = next(interface_generator)
            make_root()
        except StopIteration:
            pass

    def on_key_event(event):
        if event.key in '1234560':
            update(event.key)
        key_press_handler(event, canvas, toolbar)

    def _quit():
        nonlocal root
        root.quit()     # stops mainloop
        root.destroy()  # this is necessary on Windows to prevent
                        # Fatal Python Error: PyEval_RestoreThread: NULL tstate
        root = None

    def add_button(root, value, text):
        def h():
            update(value)

        button = Tk.Button(master=root, text="{value} - {text}".format(value=value, text=text), command=h)
        button.pack(side=Tk.LEFT)
        return h

    def make_root():
        nonlocal root, canvas, toolbar

        root = Tk.Tk()
        root.wm_title("Markup hockey player")

        # a tk.DrawingArea
        canvas = FigureCanvasTkAgg(f, master=root)
        canvas.show()
        canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

        toolbar = NavigationToolbar2TkAgg(canvas, root)
        toolbar.update()
        canvas._tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

        canvas.mpl_connect('key_press_event', on_key_event)

        button = Tk.Button(master=root, text='Quit', command=_quit)
        button.pack(side=Tk.BOTTOM)
        add_button(root, '1', 'Red team')
        add_button(root, '2', 'White team')
        add_button(root, '3', 'Orbiter')
        add_button(root, '4', 'Intersection')
        add_button(root, '5', 'Operator')
        add_button(root, '0', 'Non-person')

    make_root()

    while root is not None:
        Tk.mainloop()

def main():
    if len(sys.argv) != 2:
        print("usage: {command} hockey_dir".format(command=sys.argv[0]))
        sys.exit(1)

    hockey_dir = sys.argv[1]

    user_interface(hockey_dir, markup)

if __name__ == '__main__':
    main()
