import os
import cv2
import tqdm
import open3d as o3d
import numpy as np

from scipy.ndimage.morphology import binary_dilation
from scipy.ndimage.filters import gaussian_filter1d, median_filter, maximum_filter1d, minimum_filter1d

from top_bottom_moseq.util import interpolate, points_2d_to_3d, rescale_ir, load_matched_frames
from top_bottom_moseq.io import count_frames, videoReader, videoWriter



from ctypes import c_void_p, c_double, c_int, CDLL, cdll
from numpy.ctypeslib import ndpointer, as_ctypes
from sysconfig import get_config_var
ext_suffix = get_config_var('EXT_SUFFIX') 
lib_path = os.path.join(os.path.dirname(__file__),'..','orthographic'+ext_suffix)
_rasterize_trimesh = cdll.LoadLibrary(lib_path).rasterize_trimesh

_rasterize_trimesh.argtypes = [np.ctypeslib.ndpointer(dtype=np.float64),
                               np.ctypeslib.ndpointer(dtype=np.float64),
                               np.ctypeslib.ndpointer(dtype=np.float64),
                               np.ctypeslib.ndpointer(dtype=np.int32),
                               c_int, c_int, c_int, c_int]

def rasterize_trimesh(features, triangles, xyz, center, crop_size):
    raster = np.zeros((crop_size,crop_size,features.shape[1]), dtype=float)
    xyz_use = xyz + np.array([crop_size//2-center[0],crop_size//2-center[1],0])
    triangles_use = triangles.astype(np.int32)
    _rasterize_trimesh(raster, features, xyz_use,triangles_use,
                       crop_size, features.shape[0], features.shape[1], triangles.shape[0])
    return raster

def exclude_outliers(xyz):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd, ind1 = pcd.remove_statistical_outlier(nb_neighbors=100, std_ratio=2.0)
    if len(ind1)==0: 
        return []
    else:
        pcd, ind2 = pcd.remove_radius_outlier(nb_points=16, radius=10)
        if len(ind2)==0:
            return np.zeros(xyz.shape[0])>0
        else:
            ind = np.array(ind1)[np.array(ind2)]
            return np.in1d(np.arange(xyz.shape[0]), ind)
        
def get_ortho_data(ir, depth, mouse_mask, occl_mask, R, t, 
                   intrinsics, min_points, min_floor_depth):
    shape = ir.shape
    ir = rescale_ir(ir)
    depth = depth.astype(float)
    occl_mask = binary_dilation(occl_mask, iterations=5)
    v,u = np.array(mouse_mask[1:-1,1:-1].nonzero())+1
    
    if len(v)>min_points: 
        d,ir,mouse,occl = depth[v,u],ir[v,u],mouse_mask[v,u],occl_mask[v,u] 
        xyz = points_2d_to_3d(np.vstack((u,v,d)).T, intrinsics).dot(R.T) + t
        mask = np.all([xyz[:,2]>min_floor_depth, exclude_outliers(xyz)],axis=0)
    else: return [],np.array([]).reshape(0,3),[],[],[]

    if mask.sum()>min_points:
        v,u,ir,mouse,occl,xyz = [x[mask] for x in [v,u,ir,mouse,occl,xyz]]
        triangles = get_triangles(v,u,shape)
        return triangles,xyz,ir,mouse,occl
    else: return [],np.array([]).reshape(0,3),[],[],[]
    


def get_triangles(v,u,shape):
    index_array = np.ones(shape,dtype=int)*(-1)
    index_array[v,u] = np.arange(len(v))
    triangles = np.hstack((np.vstack((index_array[v,u],index_array[v-1,u],index_array[v,u-1],-np.ones(len(v),dtype=int))),
                           np.vstack((index_array[v,u],index_array[v+1,u],index_array[v,u+1],-np.ones(len(v),dtype=int))),
                           np.vstack((index_array[v,u],index_array[v+1,u],index_array[v,u-1],index_array[v+1,u-1])),
                           np.vstack((index_array[v,u],index_array[v-1,u],index_array[v,u+1],index_array[v-1,u+1]))))
    return triangles[:3,np.all([triangles[0,:]>=0,triangles[1,:]>=0,triangles[2,:]>=0,triangles[3,:]<0],axis=0)].T


def get_hidden_regions(rasters,shadow_surfaces):
    footprint = np.any([np.all([shadow_surfaces[0]>shadow_surfaces[1], shadow_surfaces[1]>0], axis=0), rasters[0][:,:,0]>0, rasters[1][:,:,0]>0],axis=0)
    footprint = cv2.GaussianBlur(footprint.astype(np.uint8)*255, (7,7),2)>100
    return [np.all([footprint, r[:,:,0]==0],axis=0) for r in rasters]

def get_crop_centers(prefix, camera_names, transforms, intrinsics, matched_frames):
    crop_centers = []
    for name,frames in zip(camera_names, matched_frames.T):
        with videoReader(prefix+'.'+name+'.mouse_mask.avi')    as mask_reader, \
             videoReader(prefix+'.'+name+'.depth.avi', frames) as depth_reader:

            crop_center_uvd = np.zeros((len(frames),3))*np.nan
            
            for ix,(depth,mask) in tqdm.tqdm(
                enumerate(zip(depth_reader,mask_reader)),
                desc='getting crop centers: '+name):
                
                if mask.max() > 0: 
                    crop_center_uvd[ix,:2] = np.median(mask.nonzero(),axis=1)[::-1]
                    crop_center_uvd[ix,2] = np.median(depth[np.all([mask>0,depth>0],axis=0)])

        R,t = transforms[name]
        uvd = interpolate(crop_center_uvd, axis=0)
        xyz = points_2d_to_3d(uvd, intrinsics[name]).dot(R.T)+t
        xyz = gaussian_filter1d(median_filter(xyz,(3,1)),2,axis=0)
        crop_centers.append(xyz[:,:2])    
    return np.mean(crop_centers,axis=0)


def get_raster_and_shadow(triangles, xyz, ir, mouse, occl, R, t, center, 
                          camera_name, crop_size, angle_resolution, min_floor_depth):
    

        
    if xyz.shape[0]>0: 
        if camera_name=='top': triangles = triangles[np.argsort(xyz[triangles[:,0],2])]
        if camera_name=='bottom': triangles = triangles[np.argsort(xyz[triangles[:,0],2])[::-1]]
        features = np.hstack((xyz[:,2][:,None],ir[:,None],mouse[:,None],occl[:,None]))
        raster = rasterize_trimesh(features,triangles,xyz,center,crop_size)
        raster[:,:,0] = raster[:,:,0] - min_floor_depth*(raster[:,:,0] != 0)
        raster = np.round(np.clip(raster,0,255),0)
        
        shadow_surface = get_shadow_surface(
            xyz, t, center, (0,255), angle_resolution, crop_size, camera_name
        )*(raster[:,:,0] == 0)
        
        return raster, shadow_surface
    
    else: 
        return (np.zeros((crop_size,crop_size,4)),
                np.zeros((crop_size,crop_size)))

def get_shadow_surface(xyz, camera_origin, center, clipping_bounds, ANGLE_RES, CROPSIZE, camera_name):
    dxyz = xyz - camera_origin
    angle = np.arctan2(dxyz[:,0],-dxyz[:,1])*ANGLE_RES/(2*np.pi)
    dist = np.sqrt(np.sum(dxyz[:,:2]**2,axis=1))
    slope = dxyz[:,2] / (dist+1e-3)
    shadow_slope = np.zeros(ANGLE_RES)
    shadow_start = np.zeros(ANGLE_RES)
    o = np.argsort(angle)
    ixs = np.searchsorted(angle[o],np.arange(ANGLE_RES))
    for ii,(i,j) in enumerate(zip(ixs[:-1],ixs[1:])):
        if j>i and j<len(o): 
            if camera_name=='top':
                shadow_slope[ii] = slope[o[i:j]].max()
                shadow_start[ii] = dist[o[i:j]][np.argmax(slope[o[i:j]])]
            if camera_name=='bottom':
                shadow_slope[ii] = slope[o[i:j]].min()
                shadow_start[ii] = dist[o[i:j]][np.argmin(slope[o[i:j]])]
                
    if camera_name=='top':    shadow_slope[shadow_slope != 0] = maximum_filter1d(shadow_slope[shadow_slope != 0],5)
    if camera_name=='bottom': shadow_slope[shadow_slope != 0] = minimum_filter1d(shadow_slope[shadow_slope != 0],5)
    shadow_start[shadow_slope != 0] = minimum_filter1d(shadow_start[shadow_slope != 0],5)

    x = np.arange(-CROPSIZE//2,CROPSIZE//2)[None,:]+center[0]-camera_origin[0]
    y = np.arange(-CROPSIZE//2,CROPSIZE//2)[:,None]+center[1]-camera_origin[1]
    angle = np.minimum(np.round((np.arctan(y/x) + np.pi*(x>0) + np.pi/2)*ANGLE_RES/(2*np.pi),1).astype(int),999)
    dist = np.sqrt(x**2+y**2)
    shadow_surface = (shadow_slope[angle]*dist+camera_origin[2])*(dist>=shadow_start[angle])*(shadow_slope[angle] != 0)
    return np.clip(shadow_surface, *clipping_bounds)


def orthographic_reprojection(prefix, transforms, intrinsics,
                              angle_resolution=1000, min_points=10, 
                              min_floor_depth=-5, crop_size=192,
                              overwrite_crop_centers=True):
    
    camera_names = ['top','bottom']
    matched_frames = load_matched_frames(prefix, camera_names)
    
    if os.path.exists(prefix+'.crop_centers.npy') and not overwrite_crop_centers:
        crop_centers = np.load(prefix+'.crop_centers.npy')
    else:
        crop_centers = get_crop_centers(prefix, camera_names, transforms, intrinsics, matched_frames)
        np.save(prefix+'.crop_centers.npy', crop_centers)
    
    top_frames,bottom_frames = load_matched_frames(prefix, ['top','bottom']).T
    with videoReader(prefix+'.top.ir.avi', top_frames)          as top_ir_reader, \
         videoReader(prefix+'.top.depth.avi', top_frames)       as top_depth_reader, \
         videoReader(prefix+'.top.mouse_mask.avi')              as top_mouse_mask_reader, \
         videoReader(prefix+'.top.occl_mask.avi')               as top_occl_mask_reader, \
         videoReader(prefix+'.bottom.ir.avi', bottom_frames)    as bottom_ir_reader, \
         videoReader(prefix+'.bottom.depth.avi', bottom_frames) as bottom_depth_reader, \
         videoReader(prefix+'.bottom.mouse_mask.avi')           as bottom_mouse_mask_reader, \
         videoReader(prefix+'.bottom.occl_mask.avi')            as bottom_occl_mask_reader, \
         videoWriter(prefix+'.top.ir_ortho.avi')                as top_ir_writer, \
         videoWriter(prefix+'.top.depth_ortho.avi')             as top_depth_writer, \
         videoWriter(prefix+'.top.occl_ortho.avi')              as top_occl_writer, \
         videoWriter(prefix+'.top.missing_ortho.avi')           as top_missing_writer, \
         videoWriter(prefix+'.bottom.ir_ortho.avi')             as bottom_ir_writer, \
         videoWriter(prefix+'.bottom.depth_ortho.avi')          as bottom_depth_writer, \
         videoWriter(prefix+'.bottom.occl_ortho.avi')           as bottom_occl_writer, \
         videoWriter(prefix+'.bottom.missing_ortho.avi')        as bottom_missing_writer:

        for (crop_center, top_data, bottom_data) in tqdm.tqdm(zip(crop_centers,
            zip(top_ir_reader,    top_depth_reader,    top_mouse_mask_reader,    top_occl_mask_reader),
            zip(bottom_ir_reader, bottom_depth_reader, bottom_mouse_mask_reader, bottom_occl_mask_reader)),
            desc='orthographic reprojection'):

            raster_and_shadows = []
            for camera_name,data in zip(['top','bottom'],[top_data, bottom_data]):
                ortho_data = get_ortho_data(*data, *transforms[camera_name], 
                    intrinsics[camera_name], min_points, min_floor_depth)
                raster_and_shadows.append(get_raster_and_shadow(*ortho_data, *transforms[camera_name],
                    crop_center, camera_name, crop_size, angle_resolution, min_floor_depth))

            rasters,shadows = zip(*raster_and_shadows)
            hidden_regions = get_hidden_regions(rasters,shadows)

            top_depth_writer.append( rasters[0][:,:,0].astype(np.uint8))
            top_ir_writer.append(    rasters[0][:,:,1].astype(np.uint8))
            top_occl_writer.append(  rasters[0][:,:,3].astype(np.uint8))
            top_missing_writer.append(hidden_regions[0].astype(np.uint8))

            bottom_depth_writer.append( rasters[1][:,:,0].astype(np.uint8))
            bottom_ir_writer.append(    rasters[1][:,:,1].astype(np.uint8))
            bottom_occl_writer.append(  rasters[1][:,:,3].astype(np.uint8))
            bottom_missing_writer.append(hidden_regions[1].astype(np.uint8))