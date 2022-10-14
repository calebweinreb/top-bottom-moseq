import numpy as np
import matplotlib.pyplot as plt
import tqdm
import pickle
import os
import click

from top_bottom_moseq.orthographic import *
from top_bottom_moseq.util import load_matched_frames, load_intrinsics, load_config
from top_bottom_moseq.segmentation import segment_session
from top_bottom_moseq.inpainting import inpaint_session
from top_bottom_moseq.dim_reduction import encode_session
from top_bottom_moseq.qc import save_qc_movie, grab_qc_frame

@click.command()
@click.argument('prefix')
@click.argument('config_filepath')
@click.option('--qc-vid', is_flag=True, default=True, show_default=True)
@click.option('--qc-vid-len', default=3000, show_default=True)
def main(prefix, config_filepath, qc_vid, qc_vid_len):

    # load config from yaml
    print('Loading config file')
    config = load_config(config_filepath)

    # prep data
    print('Reading intrinsics and camera transforms')
    intrinsics = {name: load_intrinsics(config['intrinsics_prefix'] + '.' + name + '.json') for name in ['top', 'bottom']}
    transforms = pickle.load(open(config['transforms_path'], 'rb'))

    # run the pipeline
    segment_session(prefix, config['mouse_segmentation_weights'], config['occlusion_segmentation_weights'])  
    orthographic_reprojection(prefix, transforms, intrinsics)
    inpaint_session(prefix, config['inpainting_weights'])
    encode_session(prefix, config['autoencoder_weights'], config['localization_weights'])

    if qc_vid:
        save_qc_movie(prefix, qc_vid_len)

if __name__ == '__main__':
    main()