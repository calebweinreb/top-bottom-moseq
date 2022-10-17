import numpy as np
import matplotlib.pyplot as plt
import tqdm
import pickle
import os
from os.path import join, exists
import click

import top_bottom_moseq
from top_bottom_moseq.orthographic import *
from top_bottom_moseq.util import load_matched_frames, load_intrinsics, load_config
from top_bottom_moseq.segmentation import segment_session
from top_bottom_moseq.inpainting import inpaint_session
from top_bottom_moseq.dim_reduction import encode_session
from top_bottom_moseq.qc import save_qc_movie, grab_qc_frame

import pdb

PATH = top_bottom_moseq.__path__._path[0]

@click.command()
@click.argument('prefix')
@click.option('--config-filepath', default='default')
@click.option('--calibn-file', default=None)
@click.option('--qc-vid', is_flag=True, default=True, show_default=True)
@click.option('--qc-vid-len', default=3000, show_default=True)
def main(prefix, config_filepath, calibn_file, qc_vid, qc_vid_len):

    # load config from yaml
    print('Loading config file')
    if config_filepath == 'default':
        config = load_config(join(PATH, 'config/default_config.yaml'))
    else:
        config = load_config(config_filepath)
        
    # Validate camera transform passed in
    if calibn_file is None and (('transforms_path' not in config) or (config['transforms_path'] is None)):
        raise ValueError('Must set path to cam params (calibration output), or provide in config file, but neither found')
    if calibn_file is not None:
        print(f'Using camera params at {calibn_file}')
    else:
        calibn_file = config['transforms_path']
        print(f'Using camera params at {calibn_file}')

    # prep data
    print('Reading intrinsics and camera transforms')
    intrinsics = {name: load_intrinsics(config['intrinsics_prefix'] + '.' + name + '.json') for name in ['top', 'bottom']}
    transforms = pickle.load(open(calibn_file, 'rb'))

    # run the pipeline
    segment_session(prefix, join(PATH, config['mouse_segmentation_weights']), join(PATH, config['occlusion_segmentation_weights']))  
    orthographic_reprojection(prefix, transforms, intrinsics)
    inpaint_session(prefix, join(PATH, config['inpainting_weights']))
    encode_session(prefix, join(PATH, config['autoencoder_weights']), join(PATH, config['localization_weights']))

    if qc_vid:
        save_qc_movie(prefix, qc_vid_len)

if __name__ == '__main__':
    main()