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
@click.option('--qc-vid/--no-qc-vid', default=True, show_default=True)
@click.option('--qc-vid-len', default=3000, show_default=True)
@click.option('--overwrite', is_flag=True, default=False, show_default=True)
@click.option('-o', '--output-dir', default=None, help='Directory into which to place processed data. Creates one folder per prefix in this parent output folder; folder name is same as the folder within which the prefix files live. Assumes one prefix per folder. (eg /path/to/20220101_data/foo.txt becomes /output/dir/20220101_data/foo.txt))')
@click.option('--steps-to-run', default='soie', help='Which processing steps to run, abbreviated to their first letter (Segment, Orthog, Inpaint, Encode)')
def main(prefix, config_filepath, calibn_file, qc_vid, qc_vid_len, overwrite, output_dir, steps_to_run):

    # Figure out which steps of the pipeline to run
    run_seg = False
    run_ortho = False
    run_inpaint = False
    run_encode = False
    if 's' in steps_to_run:
        run_seg = True
    if 'o' in steps_to_run:
        run_ortho = True
    if 'i' in steps_to_run:
        run_inpaint = True
    if 'e' in steps_to_run:
        run_encode = True

    # Update prefix to reflect new output dir if necessary
    if output_dir is not None:
        file_prefix = os.path.basename(prefix)
        parent_dir = os.path.split(os.path.split(prefix)[0])[1]
        output_dir =join(output_dir, parent_dir)
        output_prefix = join(output_dir, file_prefix)
        if not exists(output_dir):
            os.makedirs(output_dir)
    else:
        output_prefix = None

    # load config from yaml
    print('Loading config file')
    if config_filepath == 'default':
        # default config has paths relative to this package, update them here
        config = load_config(join(PATH, 'config/default_config.yaml'))
        config = {key: join(PATH, rel_path) for key, rel_path in config.items()}
    else:
        # otherwise expect absolute paths, and assume non-abs paths are defaults
        config = load_config(config_filepath)
        config = {key: join(PATH, p) if not os.path.isabs(p) else p for key, p in config.items()}

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
    if run_seg:
        segment_session(
            prefix,
            config['mouse_segmentation_weights'],
            config['occlusion_segmentation_weights'], 
            overwrite=overwrite,
            output_prefix=output_prefix)  
    else:
        print('Skipping segmentation')

    if run_ortho:
        orthographic_reprojection(
            prefix, 
            transforms, 
            intrinsics, 
            overwrite=overwrite,
            output_prefix=output_prefix)
    else:
        print('Skipping orthographic reproj.')
    
    if run_inpaint:
        inpaint_session(
            prefix, 
            config['inpainting_weights'], 
            output_prefix=output_prefix,
            overwrite=overwrite)
    else:
        print('Skipping inpainting')

    if run_encode:
        encode_session(
            prefix, 
            config['autoencoder_weights'], 
            config['localization_weights'], 
            output_prefix=output_prefix,
            overwrite=overwrite)
    else:
        print('Skipping encoding')

    if qc_vid:
        save_qc_movie(prefix, qc_vid_len, output_prefix=output_prefix)

if __name__ == '__main__':
    main()