#!/usr/bin/env python3

import sys
import os.path
import itertools

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

from skimage import filters, segmentation, measure, io, morphology

USER_INFO = """Classes:
    1 - Red team
    2 - White team
    3 - Orbiter
    4 - Double person
    0 - Non-person
"""

START_SAMPLE = 45

def get_filepath(source_dir, filename):
    return os.path.abspath(os.path.join(os.path.expanduser(source_dir), filename))

def markup(hockey_dir, ax):
    class Updater:
        value = None
    updater = Updater()

    print(USER_INFO)

    image_dir = get_filepath(hockey_dir, 'images')
    sample_num = 0
    sample_dir = get_filepath(hockey_dir, 'sample')
    os.makedirs(sample_dir, exist_ok=True)

    with open(get_filepath(image_dir, 'gt.txt'), 'a') as gt:
        for i in itertools.count(0, 10):
            try:
                frame_filename = 'frame_{i}.png'.format(i=i)
                frame = io.imread(get_filepath(image_dir, frame_filename))
                mask = io.imread(get_filepath(image_dir, 'mask_{i}.png'.format(i=i)))

            except FileNotFoundError:
                break

            # apply threshold
            thresh = filters.threshold_otsu(mask)
            bw = morphology.closing(mask > thresh, morphology.square(3))

            # remove artifacts connected to image border
            cleared = bw.copy()
            segmentation.clear_border(cleared)

            # label image regions
            label_image = measure.label(cleared)

            for region in measure.regionprops(label_image):
                if region.area < 250:
                    continue

                if sample_num >= START_SAMPLE:
                    # draw rectangle around segmented coins
                    minr, minc, maxr, maxc = region.bbox

                    sample = frame[minr:maxr, minc:maxc]
                    sample_mask = cleared[minr:maxr, minc:maxc]
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
        if event.key in '123450':
            update(event.key)
        key_press_handler(event, canvas, toolbar)

    def _quit():
        root.quit()     # stops mainloop
        root.destroy()  # this is necessary on Windows to prevent
                        # Fatal Python Error: PyEval_RestoreThread: NULL tstate

    def add_button(root, value, text):
        def h():
            update(value)

        button = Tk.Button(master=root, text=text, command=h)
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
        add_button(root, '4', 'Double player')
        add_button(root, '0', 'Non-person')
        Tk.mainloop()

    make_root()

def main():
    if len(sys.argv) != 2:
        print("usage: {command} hockey_dir".format(command=sys.argv[0]))
        sys.exit(1)

    hockey_dir = sys.argv[1]

    user_interface(hockey_dir, markup)

if __name__ == '__main__':
    main()
