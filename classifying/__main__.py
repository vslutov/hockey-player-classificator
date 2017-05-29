#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ----------------------------------------------------------------------------
# "THE BEER-WARE LICENSE" (Revision 42):
# <vslutov@yandex.ru> wrote this file.   As long as you retain this notice you
# can do whatever you want with this stuff. If we meet some day, and you think
# this stuff is worth it, you can buy me a beer in return.      Vladimir Lutov
# ----------------------------------------------------------------------------

import argparse

from markup import user_interface, markup, update_samples
from classify import classify
from validate import validate

def main():
    parser = argparse.ArgumentParser(description='Hockey player classificator')
    parser.add_argument('command', choices=['markup', 'update_samples',
                                            'classify', 'validate'])
    parser.add_argument('hockey_dir')

    args = parser.parse_args()

    if args.command == 'markup':
        user_interface(args.hockey_dir, markup)
    elif args.command == 'update_samples':
        update_samples(args.hockey_dir)
    elif args.command == 'classify':
        classify(args.hockey_dir)
    elif args.command == 'validate':
        validate(args.hockey_dir)

if __name__ == '__main__':
    main()
