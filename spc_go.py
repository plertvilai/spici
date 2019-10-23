#!/usr/bin/env python3
import argparse
import sys
# allow importing from parent dir
sys.path.append('..')
from spici.spcserver import SPCServer

def parse_cmds():
    parser = argparse.ArgumentParser(description='Accessing spc.ucsd.edu pipeline')
    parser.add_argument('--search-param-file', default=None, help='spc.ucsd.edu search param path')
    parser.add_argument('--image-output-path', default=None, help='Downloaded images output path')
    parser.add_argument('--meta-output-path', default=None, help='Meta data output path')
    parser.add_argument('--daylight_savings', default=True, help='Daylight savings option', action='store_false')
    parser.add_argument('-d', '--download', default=False, help='Download flagging option', action='store_true')
    parser.add_argument('-u', '--upload', default=False, help='Download flagging option', action='store_true')
    args = parser.parse_args(sys.argv[1:])
    return args

def main(args):
    print("Downloading images...")
    spc = SPCServer(daylight_savings=args.daylight_savings)
    spc.retrieve(textfile=args.search_param_file,
                 output_dir=args.image_output_path,
                 output_csv_filename=args.meta_output_path,
                 download=args.download)

if __name__ == '__main__':
    main(parse_cmds())



