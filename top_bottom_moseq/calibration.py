import numpy as np
import tqdm
import cv2
import pickle
import imageio
import matplotlib.pyplot as plt

from top_bottom_moseq.io import videoReader
from top_bottom_moseq.util import rescale_ir
from functools import partial
from os.path import join, exists

def find_ir_corners(packed_tuple,
 cv2_chkb_flags=None,
 criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
 checker_dims=(8,6)):

    # Unpack data
    ix, (ir_frame, depth_frame) = packed_tuple
    ir_frame = rescale_ir(ir_frame).astype(np.uint8)

    # Do analysis
    ret, corners_approx = cv2.findChessboardCorners(ir_frame, checker_dims, None, cv2_chkb_flags)

    corner = None
    if ret:
        uv = cv2.cornerSubPix(ir_frame, corners_approx, (5, 5), (-1, -1), criteria).squeeze()
        d = depth_frame[uv[:,1].astype(int),uv[:,0].astype(int)]
        if np.all(d>0):
            corner = np.hstack((uv,d[:,None]))
    return corner, ix

def detect_corners_from_video(prefix, checker_dims, overwrite=False, cv2_chkb_flags=None, corners_suffix='corners', parallel=False):
    """# Detect corners from a checkerboard calibration video.    

    Arguments:
        prefix {str} -- prefix to top-bottom files
        checker_dims {tuple} -- dimensions of inner intersections of calibration checkerboard 

    Keyword Arguments:
        overwrite {bool} -- if true, ignore existing output and overwrite it (default: {False})
        cv2_chkb_flags {[type]} -- cv2 flags for checkerboard detection (default: {None})
        corners_suffix {str} -- file suffix; can use if debugging various methods of corner detection (default: {'corners'})
        parallel {bool} -- whether to run in parallel. Should see speed up for 4+ CPUs. (default: {False})

    Returns:
        corners -- 
        ixs -- timestamps of all detected corners
    """
    corners_fname = join(prefix + f'.{corners_suffix}.p')

    # Load data if it already exists
    if exists(corners_fname) and not overwrite:
        print('Corners already detected, re-loading...')
        with open(corners_fname, 'rb') as f:
            d = pickle.load(f)
        corners = d['corners']
        ixs = d['ixs']
        return corners, ixs
    
    # Otherwise process the video
    if parallel:
        from tqdm.contrib.concurrent import process_map
        func = partial(find_ir_corners, checker_dims=checker_dims, cv2_chkb_flags=cv2_chkb_flags)
        with videoReader(prefix+'.ir.avi') as ir_reader, videoReader(prefix+'.depth.avi') as depth_reader:
            out = process_map(func, enumerate(zip(ir_reader, depth_reader)))
        corners = [tup[0] for tup in out if tup[0] is not None]
        ixs = [tup[1] for tup in out if tup[0] is not None]
    else:
        corners = [] # store the output as a list of (n,3) arrays in pixel-by-depth space
        ixs     = [] # also return the frame indexes where points were detected
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        with videoReader(prefix+'.ir.avi') as ir_reader, videoReader(prefix+'.depth.avi') as depth_reader:
            for ix,(ir,depth) in tqdm.tqdm(enumerate(zip(ir_reader,depth_reader))):
                ir = rescale_ir(ir).astype(np.uint8)
                ret, corners_approx = cv2.findChessboardCorners(ir, checker_dims, None, cv2_chkb_flags)
                if ret:
                    uv = cv2.cornerSubPix(ir, corners_approx, (5, 5), (-1, -1), criteria).squeeze()
                    d = depth[uv[:,1].astype(int),uv[:,0].astype(int)]
                    if np.all(d>0):
                        corners.append(np.hstack((uv,d[:,None])))
                        ixs.append(ix)

    # Save the output
    with open(corners_fname, 'wb') as f:
        d = pickle.dump({'corners': corners, 'ixs': ixs}, f)

    return corners, ixs
    



def rigid_transform_3D(A, B):
    N = A.shape[0]; # total points
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    
    # centre the points
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))

    # dot is matrix multiplication for array
    H = np.transpose(AA).dot(BB)
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T.dot(U.T)
    
    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[2,:] *= -1
        R = Vt.T.dot(U.T)

    t = -R.dot(centroid_A.T) + centroid_B.T
    return R, t

def get_corner_label_reindexes(checker_dims):
    corner_label_matrix = np.arange(checker_dims[0]*checker_dims[1]).reshape(checker_dims[::-1])
    return [
        corner_label_matrix.flatten(),
        corner_label_matrix.flatten()[::-1],
        corner_label_matrix[::-1,:].flatten(),
        corner_label_matrix[::-1,:].flatten()[::-1],
        corner_label_matrix.T.flatten(),
        corner_label_matrix.T.flatten()[::-1],
        corner_label_matrix.T[::-1,:].flatten(),
        corner_label_matrix.T[::-1,:].flatten()[::-1]
    ],[
        'identity',
        'reverse',
        'flip',
        'flip+reverse',
        'transpose',
        'transpose+reverse',
        'transpose+flip',
        'transpose+flip+reverse',
    ]

def save_corner_detection_video(calibration_prefix, camera, linewidth=1, radius=5, overwrite=False):
    corner_data = pickle.load(open(calibration_prefix+'.'+camera+'.corners.p','rb'))
    corner_ixs = np.array(corner_data['ixs'])
    save_path = calibration_prefix+'.'+camera+'.corners.mp4'

    if exists(save_path) and not overwrite:
        print('Corners video already written...')
        return

    with videoReader(calibration_prefix+'.'+camera+'.ir.avi') as reader, \
        imageio.get_writer(save_path, pixelformat='yuv420p', fps=30, quality=6) as writer:
        for ix,im in tqdm.tqdm(enumerate(reader), desc='Corner detection video'):
            im = np.repeat(rescale_ir(im)[:,:,None],3,axis=2).astype(np.uint8)
            corner_ix = corner_ixs.searchsorted(ix)
            if corner_ix >= corner_ixs.shape:  
                continue  # ie, if no more corners detected in the video
            elif ix==corner_ixs[corner_ix]:
                uvs = [(int(u),int(v)) for u,v in corner_data['corners'][corner_ix][:,:2]]
                colors = [tuple([int(c*255) for c in plt.cm.viridis(x)[:3]]) for x in np.linspace(0,1,len(uvs))]
                for uv1,uv2 in zip(uvs[:-1],uvs[1:]):
                    im = cv2.line(im, uv1, uv2, (0,0,0), linewidth, cv2.LINE_AA)
                for uv,c in zip(uvs,colors):
                    im = cv2.circle(im, uv, radius, c, -1, cv2.LINE_AA)
            text = '{} ({})'.format(ix, corner_ix)
            im = cv2.putText(im, text, (10,im.shape[0]-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1,cv2.LINE_AA)
            writer.append_data(im)