import numpy as np
import matplotlib.pyplot as plt
import tqdm
import pickle
from os.path import join, exists
from glob import glob
import click
import pdb

import top_bottom_moseq
from top_bottom_moseq.util import load_matched_frames, points_2d_to_3d, load_intrinsics, load_config, check_if_already_done
from top_bottom_moseq.calibration import *

PATH = top_bottom_moseq.__path__._path[0]

@click.command()
@click.argument('base_path')
@click.option('--config-filepath', default='default')
@click.option('--out_path', default='.')
@click.option('--checker_dims', nargs=2, default=(8,6))
@click.option('--parallel', is_flag=True, default=False)
@click.option('--qc-vid', is_flag=True, default=True)
@click.option('--overwrite', is_flag=True, default=False)
def main(base_path, config_filepath, out_path, checker_dims, parallel, qc_vid, overwrite):
    """Process calibration videos stored at base_path

    Arguments:
        base_path {str} -- path to folder containing flat and slant calibration videos (each respectively nested in folders containing the words "flat" and "slant")
        out_path {str} -- relative path, from base_path, of where to put the output figs and transform pickle
        checker_dims {tuple} -- dims of checkerboard for cv2
        parallel {bool} -- whether to use parallel processing. Should see dramatic speed ups.
        qc_vid {bool} -- whether to write a QC video (doesn't take too long)
    """
    camera_names = ['top', 'bottom']
    out_path = join(base_path, out_path)

    # Load configs
    print('Loading config file')
    if config_filepath == 'default':
        config = load_config(join(PATH, 'config/default_config.yaml'))
    else:
        config = load_config(config_filepath)
    intrinsics_prefix = config['intrinsics_prefix']

    if parallel:
        print('Using parallel processing...')

    # Find calibration files for this session
    flat_prefix = glob(join(base_path, '*flat*/*.avi'))[0].split('.')[0]
    slant_prefix = glob(join(base_path, '*slant*/*.avi'))[0].split('.')[0]

    calibration_points = []
    for name in camera_names:
        corners, ixs = detect_corners_from_video(slant_prefix + '.' + name, checker_dims, parallel=parallel, overwrite=overwrite)
        assert len(corners)>0, 'No checker points detected'
        intrinsics = load_intrinsics(intrinsics_prefix + '.' + name + '.json')
        corners_3d = np.array([points_2d_to_3d(points,intrinsics) for points in corners])
        
        
        # quality control: each frame should have the same set of distances between points
        # so for each point-set, find the distances to the first point
        # get the median of these distances across all point-sets
        # then exclude point-sets where the rmsd to the median exceeds 2mm
        distances = np.array([np.sqrt(((points-points[0,:])**2).sum(1)) for points in corners_3d])
        rmsd = np.mean((np.median(distances,axis=0)-distances)**2,axis=1)
        calibration_points.append({ix:points for ix,points,error in zip(ixs,corners_3d,rmsd) if error<2})
        
        plt.figure()
        plt.plot(rmsd)
        plt.axhline(2, c='k')
        plt.ylim([-1,20])
        plt.gcf().set_size_inches((6,2))
        plt.tight_layout()
        plt.savefig(join(out_path, f'within_{name}_cam_rmsd.png'))

        if qc_vid:
            save_corner_detection_video(slant_prefix, name, overwrite=overwrite)

    # Align timestamps and calculate projection params¶
    # Filter to frames where calibration points were detected for all cameras
    matched_timestamps = load_matched_frames(slant_prefix, camera_names)
    filtered_matched_timestamps = np.array([ixs for ixs in matched_timestamps if np.all([ix in ps for ix,ps in zip(ixs,calibration_points)])])
    matched_calib_points = [np.array([ps[ix] for ix in filtered_matched_timestamps[:,i]]) for i,ps in enumerate(calibration_points)]

    reindexes, names  = get_corner_label_reindexes(checker_dims)
    projection_params = []
    for calib_points in matched_calib_points[:-1]:
        param_for_each_reindex = []
        error_for_each_reindex = [] 
        for reindex in reindexes:
            source_points = calib_points[:,reindex,:].reshape(-1,3)
            target_points = matched_calib_points[-1].reshape(-1,3)
            R,t = rigid_transform_3D(source_points, target_points)
            error = np.mean((source_points.dot(R.T)+t - target_points)**2)
            param_for_each_reindex.append((R,t))
            error_for_each_reindex.append(round(error,1)) 
        print('Errors:'+'\n   '.join(['','identity\t= {:.1f}','reverse\t= {:.1f}',
                                    'flip-vert\t= {:.1f}','flip-vert+r\t= {:.1f}']).format(*error_for_each_reindex))
        projection_params.append(param_for_each_reindex[np.argmin(error_for_each_reindex)])     



    # Check the agreement of coordinates for each checker point between the top and bottom views
    bottom_points = matched_calib_points[1].reshape(-1,3)
    top_points = matched_calib_points[0][:,reindexes[3],:].reshape(-1,3)

    R_top2bottom,t_top2bottom = projection_params[0]
    top_points_transformed = top_points.dot(R_top2bottom.T)+t_top2bottom

    fig,axs = plt.subplots(1,3)
    for i,coord_name in enumerate(['x','y','z']):
        axs[i].scatter(top_points_transformed[:,i],bottom_points[:,i],s=.1)
        axs[i].set_xlabel('Top camera (transformed)')
        axs[i].set_ylabel('Bottom camera')
        rmsd = np.sqrt(np.mean((top_points_transformed[:,i]-bottom_points[:,i])**2))
        axs[i].set_title(coord_name+'-coordinate\nRMSD = '+str(round(rmsd,2)))
    fig.subplots_adjust(wspace=0.5)
    fig.suptitle(f'{matched_timestamps.shape[0]} matched frames')
    fig.set_size_inches((10,2.5))
    plt.tight_layout()
    plt.savefig(join(out_path, 'across_cam_rmsd.png'))

    # Next find the plane of the acryclic that the mouse walks on.
    corners,ixs = detect_corners_from_video(flat_prefix + '.bottom', checker_dims)
    intrinsics = load_intrinsics(intrinsics_prefix + '.bottom.json')
    corners_3d = np.array([points_2d_to_3d(points,intrinsics) for points in corners]).reshape(-1,3)
    
    # fit plane using RANSAC from sklearn
    from sklearn.linear_model import RANSACRegressor
    ransac = RANSACRegressor(residual_threshold=12).fit(corners_3d[:,:2],corners_3d[:,2])

    # show agreement between ransac estimator and raw data
    fig,axs = plt.subplots(1,3)
    axs[0].scatter(corners_3d[:,0],corners_3d[:,1],c=corners_3d[:,2], vmin=np.percentile(corners_3d[:,2],1),vmax=np.percentile(corners_3d[:,2],99),s=2)
    axs[1].scatter(corners_3d[:,0],corners_3d[:,1],c=ransac.predict(corners_3d[:,:2]), vmin=np.percentile(corners_3d[:,2],1),vmax=np.percentile(corners_3d[:,2],99),s=2)
    axs[2].scatter(corners_3d[:,2],ransac.predict(corners_3d[:,:2]), s=.2, linewidth=0)
    axs[0].set_title('Original z-values')
    axs[1].set_title('RANSAC z-values')
    axs[2].set_xlabel('Original Z')
    axs[2].set_ylabel('RANSAC Z')
    fig.set_size_inches((10,3.5))
    plt.tight_layout()
    plt.savefig(join(out_path, 'floor_plane_regression.png'))

    # calculate the transform
    a,b = ransac.estimator_.coef_
    t = ransac.estimator_.intercept_
    source = np.array([[0,0,t],[-a,-b,t+1],[1,0,t+a]])
    target = np.array([[0,0,0],[0,0,np.sqrt(a**2+b**2+1)],[np.sqrt(1+a**2),0,0]])
    R_bottom2final, t_bottom2final = rigid_transform_3D(source,target)

    # Check that points are flat after transformation
    corners_3d_transformed = corners_3d.dot(R_bottom2final.T) + t_bottom2final
    fig,axs = plt.subplots(1,3)
    axs[0].scatter(corners_3d_transformed[:,0],corners_3d_transformed[:,1],c=corners_3d_transformed[:,2], vmax=10,s=2)
    axs[1].scatter(corners_3d_transformed[:,0],corners_3d_transformed[:,2],s=.4, linewidth=0)
    axs[2].scatter(corners_3d_transformed[:,1],corners_3d_transformed[:,2],s=.4, linewidth=0)
    fig.set_size_inches((10,3.5))
    plt.tight_layout()
    plt.savefig(join(out_path, 'final_points_crosssection.png'))

    # Compose transformations top2bottom and bottom2final to get top2final, then save
    R_top2final = R_bottom2final.dot(R_top2bottom)
    t_top2final = R_bottom2final.dot(t_top2bottom) + t_bottom2final

    
    transform_name = join(out_path, 'camera_3D_transforms.p')
    print(f'Saving final transforms to {transform_name}')
    with open(transform_name, 'wb') as out_file:
        pickle.dump({'top':    (R_top2final,t_top2final), 
                    'bottom': (R_bottom2final,t_bottom2final)},
                    out_file)


if __name__ == '__main__':
    main()
