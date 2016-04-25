#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse

from markup import user_interface, markup, update_samples
from classify import classify

def main():
    parser = argparse.ArgumentParser(description='Hockey player classificator')
    parser.add_argument('command', choices=['markup', 'update_samples', 'classify'])
    parser.add_argument('hockey_dir')

    args = parser.parse_args()

    if args.command == 'markup':
        user_interface(args.hockey_dir, markup)
    elif args.command == 'update_samples':
        update_samples(args.hockey_dir)
    elif args.command == 'classify':
        classify(args.hockey_dir)

if __name__ == '__main__':
    main()
