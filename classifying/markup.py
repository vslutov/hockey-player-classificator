#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------
# "THE BEER-WARE LICENSE" (Revision 42):
# <vslutov@yandex.ru> wrote this file.   As long as you retain this notice you
# can do whatever you want with this stuff. If we meet some day, and you think
# this stuff is worth it, you can buy me a beer in return.      Vladimir Lutov
# ----------------------------------------------------------------------------

import sys
import os.path
import glob
import itertools
import csv
import tarfile
import functools

if sys.version_info[0] < 3:
    import Tkinter as Tk
else:
    import tkinter as Tk

import numpy as np
# import cv2

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2TkAgg
from matplotlib.backend_bases import key_press_handler
from matplotlib.figure import Figure

from skimage import segmentation, measure, io, transform

BUCKET_SIZE = 10000

def get_filepath(source_dir, filename):
    return os.path.abspath(os.path.join(os.path.expanduser(source_dir), filename))

def get_label(filled_image, previous_props):
    result = -1

    for prop, label in previous_props:
        if np.logical_and(filled_image, prop).any():
            if result == -1:
                result = label
            else:
                return -1
    return result

def update_previous(previous_props, label_image):
    result = []
    background = label_image == 0

    for prop, label in previous_props:
        not_interesting = np.logical_not(prop)
        interesting = np.ma.MaskedArray(label_image, not_interesting)
        find = interesting.max()
        if np.logical_or(interesting == find, background).all():
            result.append((prop, label))

    return result

def get_sample_nums(gt_filepath):
    max_sample = -1
    with open(gt_filepath, 'r') as gt:
        previous_gt = [line for line in csv.reader(gt)]
        if len(previous_gt) > 0:
            max_sample = max(int(line[0]) for line in previous_gt)
            start_frame = previous_gt[-1][1]
            start_frame = int(start_frame[len('input'):-len('.png')])
            for i, line in enumerate(previous_gt):
                if line[1] == 'input{i}.png'.format(i=start_frame - 1):
                    start_frame -= 1
                    start_sample = i
                    break
            print(start_frame)
        else:
            max_sample, start_frame, start_sample = -1, 3600, 0

    return max_sample, start_frame, start_sample

def save_samples(hockey_dir, new_samples):
    np.save(get_filepath(hockey_dir, 'samples_{i}.npy'.format(i=new_samples[0])),
            np.array(new_samples[1]))
    new_samples[0] += 1
    new_samples[1] = []

def save_chains(hockey_dir, chains):
    with open(get_filepath(hockey_dir, 'chains.txt'), 'w') as output:
        for chain in chains:
            print(','.join(str(elem) for elem in chain), file=output)

def update_samples(hockey_dir):
    for filename in glob.iglob(get_filepath(hockey_dir, 'samples_*.npy')):
        os.remove(filename)

    video_dir = get_filepath(hockey_dir, 'work_video')
    mask_dir = get_filepath(hockey_dir, 'masks')
    video_template = get_filepath(video_dir, 'cska_akbars_cam_3_{begin}_{end}.avi')

    sample_num = 0
    new_samples = [0, []]
    chains = []
    previous_props = []

    video = cv2.VideoCapture()

    with open(get_filepath(hockey_dir, 'coords.csv'), 'w') as coords:
        for i in range(0, 165600):
            ret, frame = video.read()
            if ret is False:
                if video.open(video_template.format(begin=i, end=i + 1199)):
                    ret, frame = video.read()
                else:
                    continue

            BORDER = 80
            frame = frame[:-BORDER]
            try:
                mask = io.imread(get_filepath(mask_dir, 'mask{i}.png'.format(i=i)), 0)[:-BORDER]
            except FileNotFoundError:
                continue

            # apply threshold
            cleared = mask > 128
            cleared = segmentation.clear_border(cleared)

            # label image regions
            label_image = measure.label(cleared)
            current_props = measure.regionprops(label_image)

            previous_props = update_previous(previous_props, label_image)
            next_props = []

            for region in current_props:
                if region.area < 250:
                    continue

                minr, minc, maxr, maxc = region.bbox
                print(sample_num, i, minr, minc, maxr, maxc, sep=',', file=coords)

                filled_image = np.zeros(frame.shape[:2], dtype=bool)
                filled_image[minr:maxr, minc:maxc] = region.filled_image

                sample = transform.resize(frame[minr:maxr, minc:maxc], (64, 32))
                sample_mask = region.filled_image.astype(np.uint8) * 255
                sample_mask = transform.resize(sample_mask, (64, 32))
                sample = np.dstack((sample, sample_mask))

                new_samples[1].append(sample)

                chain_number = get_label(filled_image, previous_props)
                if chain_number == -1:
                    chain_number = len(chains)
                    chains.append([])

                chains[chain_number].append(sample_num)
                next_props.append((filled_image, chain_number))

                sample_num += 1

                if sample_num % BUCKET_SIZE == 0:
                    print('sample', sample_num, ', frame', i)
                    save_samples(hockey_dir, new_samples)
                    save_chains(hockey_dir, chains)

            previous_props = next_props


    video.release()
    save_samples(hockey_dir, new_samples)
    save_chains(hockey_dir, chains)

@functools.lru_cache(maxsize=2)
def get_bucket(hockey_dir, i):
    return np.load(get_filepath(hockey_dir, 'samples_{i}.npy'.format(i=i)))

def get_sample(hockey_dir, i):
    bucket = get_bucket(hockey_dir, i // BUCKET_SIZE)
    return bucket[i % BUCKET_SIZE][:, :, 2::-1]

def markup(hockey_dir, ax):
    class Updater:
        value = None
    updater = Updater()

    gt_filepath = get_filepath(hockey_dir, 'gt.txt')
    chains_filepath = get_filepath(hockey_dir, 'chains.txt')

    with open(chains_filepath, 'r') as chain_fd:
        chains = [[int(elem) for elem in chain.split(',')] for chain in chain_fd]

    with open(gt_filepath, 'r') as gt:
        previous_gt = {int(line[0]): line[1] for line in csv.reader(gt)}

    shape = get_sample(hockey_dir, 0).shape

    with open(gt_filepath, 'a') as gt:
        for chain in chains:
            updater.value = -1

            for elem in chain:
                if elem in previous_gt:
                    updater.value = previous_gt[elem]
                    break

            if updater.value == -1 and len(chain) >= 10:

                sample = np.zeros((shape[0] * 2, shape[1] * 2, shape[2]))
                sample[:shape[0], :shape[1]] = get_sample(hockey_dir, chain[0])
                sample[shape[0]:, :shape[1]] = get_sample(hockey_dir, chain[5])
                sample[:shape[0], shape[1]:] = get_sample(hockey_dir, chain[9])
                sample[shape[0]:, shape[1]:] = get_sample(hockey_dir, chain[-1])

                ax.cla()
                ax.imshow(sample)

                yield updater

                if updater.value == -1:
                    return
                else:
                    for elem in chain:
                        print(elem, updater.value, sep=',', file=gt)
                        previous_gt[elem] = updater.value

def user_interface(hockey_dir, interface_generator):

    root, canvas, toolbar = None, None, None

    f = Figure(figsize=(5, 4), dpi=100)
    ax = f.add_subplot(111)
    interface_generator = interface_generator(hockey_dir, ax)
    updater = next(interface_generator)

    def update(value):
        nonlocal updater
        updater.value = value
        _quit(quit=False)

        try:
            updater = next(interface_generator)
            make_root()
        except StopIteration:
            pass

    def on_key_event(event):
        if event.key in '1234560':
            update(event.key)
        key_press_handler(event, canvas, toolbar)

    def _quit(quit=True):
        nonlocal root, updater
        if quit:
            updater.value = -1
            try:
                updater = next(interface_generator)
            except StopIteration:
                pass
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
        add_button(root, '5', 'Trash')
        add_button(root, '0', 'Non-person')

    make_root()

    while root is not None:
        Tk.mainloop()
