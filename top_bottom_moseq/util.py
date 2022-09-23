import numpy as np
import cv2
import av
import json
from top_bottom_moseq.io import count_frames

def interpolate(x, axis=0):
    x = np.moveaxis(x,axis,0)
    shape = x.shape
    x = x.reshape(shape[0],-1)
    for i in range(x.shape[1]):
        x[:,i] = np.interp(
            np.arange(x.shape[0]),
            np.nonzero(~np.isnan(x[:,i]))[0],
            x[:,i][~np.isnan(x[:,i])])
    return np.moveaxis(x.reshape(shape),0,axis)


def rescale_ir(x): 
    return np.clip((np.log(np.clip(x.astype(float)+100,160,5500))-5)*70,0,255)

def load_intrinsics(param_path):
    # Load camera parameters from the json file exported by pyk4a
    for params in json.load(open(param_path,'r'))['CalibrationInformation']['Cameras']:
        if params['Location'] == 'CALIBRATION_CameraLocationD0':
            cx,cy,fx,fy,k1,k2,k3,k4,k5,k6,_,_,p2,p1 = tuple(params['Intrinsics']['ModelParameters'])
            
            # This works for NFOV_UNBINNED
            calibration_image_binned_resolution = (1024,1024)
            crop_offset = (192,180)  
            cx = cx * calibration_image_binned_resolution[0] - crop_offset[0] - 0.5
            cy = cy * calibration_image_binned_resolution[1] - crop_offset[1] - 0.5
            fx *= calibration_image_binned_resolution[0]
            fy *= calibration_image_binned_resolution[1]
            
            cameraMatrix = np.array([[fx,0,cx],[0,fy,cy],[0,0,1]])
            distCoeffs = np.array([k1,k2,p1,p2,k3,k4,k5,k6])
            return (cameraMatrix, distCoeffs)


def load_matched_frames(prefix, camera_names):
    timestamps = []
    for name in camera_names:
        num_frames = np.min([
            count_frames(prefix+'.'+name+'.ir.avi'),
            count_frames(prefix+'.'+name+'.depth.avi')])
        timestamps.append(np.load('.'.join([prefix,name,'device_timestamps.npy']))[:num_frames])
    return match_frames(timestamps)

def match_frames(timestamps, frame_length=33333):
    # input: timestamps is a list of N arrays, where each array contains the device timestamps from a synced camera
    # return: n x N array of frame indexes for each camera. Only frames captured by all N cameras are included

    # timestamps are in microseconds and should all be (approximate) multiple of 33333 + a fixed offset thats <2ms
    # 1) convert timestamps to integers representing the number of 33333us clock periods
    # 2) create a table where entry (i,j) if the frame number for camera j at clock period i (or -1 if there is no frame)
    # 3) return indexes corresponding to mask rows where (i,j) != -1 for all cameras j

    timestamps = [np.rint(ts/frame_length).astype(int) for ts in timestamps]
    mask = np.ones((np.max(np.hstack(timestamps))+1,len(timestamps)))*(-1)
    for j,ts in enumerate(timestamps): mask[ts,j] = np.arange(len(ts))
    return mask[np.all(mask > -1, axis=1),:].astype(int)

def points_2d_to_3d(points, intrinsics):
    # convert points from depthmap space to 3D space
    # points should have shape (N,3) representing coords (u,v,d)
    
    fx = intrinsics[0][0, 0]
    fy = intrinsics[0][1, 1]
    cx = intrinsics[0][0, 2]
    cy = intrinsics[0][1, 2] 
    
    d = points[:,2][:,None] # depth as column vector
    uv_undistorted = cv2.undistortPoints(points[:,None,:2].astype(float),*intrinsics).squeeze() # undistort pixel space
    points_3d = np.hstack((uv_undistorted * d, d)) # unproject points
    return points_3d

def crop(X, crop_center, crop_size):
    crop_center = crop_center.astype(int)
    crop_center = np.minimum(np.maximum(crop_center,crop_size),np.array(X.shape[-2:])-crop_size)
    return X[...,crop_center[0]-crop_size:crop_center[0]+crop_size,
                 crop_center[1]-crop_size:crop_center[1]+crop_size]

def uncrop(X_cropped, crop_center, crop_size, shape):
    X = np.zeros(shape)
    crop_center = crop_center.astype(int)
    X[...,crop_center[0]-crop_size:crop_center[0]+crop_size,
          crop_center[1]-crop_size:crop_center[1]+crop_size] = X_cropped
    return X