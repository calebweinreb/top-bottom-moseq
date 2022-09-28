import numpy as np
import tqdm
import cv2

from top_bottom_moseq.io import videoReader
from top_bottom_moseq.util import rescale_ir
from functools import partial

def find_ir_corners(packed_tuple,
 flags=None,
 criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001),
 checker_dims= (8,6)):

    # Unpack data
    ix, (ir_frame, depth_frame) = packed_tuple
    ir_frame = rescale_ir(ir_frame).astype(np.uint8)

    # Do analysis
    ret, corners_approx = cv2.findChessboardCorners(ir_frame, checker_dims, None, flags)

    corner = None
    if ret:
        uv = cv2.cornerSubPix(ir_frame, corners_approx, (5, 5), (-1, -1), criteria).squeeze()
        d = depth_frame[uv[:,1].astype(int),uv[:,0].astype(int)]
        if np.all(d>0):
            corner = np.hstack((uv,d[:,None]))
    return corner, ix

def detect_corners_from_video_parallel(prefix, checker_dims, cv2_flags=None):
    from tqdm.contrib.concurrent import process_map

    # Detect corners from a checkerboard calibration video
    # store the output as a list of (n,3) arrays in pixel-by-depth space
    # also return the frame indexes where points were detected
    
    func = partial(find_ir_corners, checker_dims=checker_dims, flags=cv2_flags)
    with videoReader(prefix+'.ir.avi') as ir_reader, videoReader(prefix+'.depth.avi') as depth_reader:
        out = process_map(func, enumerate(zip(ir_reader, depth_reader)))

    corners = [tup[0] for tup in out if tup[0] is not None]
    ixs = [tup[1] for tup in out if tup[0] is not None]

    return corners, ixs

def detect_corners_from_video(prefix, checker_dims):
    # Detect corners from a checkerboard calibration video
    corners = [] # store the output as a list of (n,3) arrays in pixel-by-depth space
    ixs     = [] # also return the frame indexes where points were detected
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    flags = (cv2.CALIB_CB_ADAPTIVE_THRESH)    
    with videoReader(prefix+'.ir.avi') as ir_reader, videoReader(prefix+'.depth.avi') as depth_reader:
        for ix,(ir,depth) in tqdm.tqdm(enumerate(zip(ir_reader,depth_reader))):
            ir = rescale_ir(ir).astype(np.uint8)
            ret, corners_approx = cv2.findChessboardCorners(ir, checker_dims, None, flags)
            if ret:
                uv = cv2.cornerSubPix(ir, corners_approx, (5, 5), (-1, -1), criteria).squeeze()
                d = depth[uv[:,1].astype(int),uv[:,0].astype(int)]
                if np.all(d>0):
                    corners.append(np.hstack((uv,d[:,None])))
                    ixs.append(ix)
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
        corner_label_matrix[::-1,:].flatten()[::-1]
    ]