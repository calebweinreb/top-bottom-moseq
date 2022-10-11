import numpy as np
import cv2
import os
import tqdm
import imageio
from itertools import product
import matplotlib.pyplot as plt

from top_bottom_moseq.io import read_frames, videoReader, videoWriter, count_frames
from top_bottom_moseq.util import rescale_ir, load_matched_frames, crop

def clip_and_color(x, low, high):
    x = np.clip((x.astype(float)-low)/(high-low),0,1)
    return (plt.cm.viridis(x)*255)[:,:,:3].astype(np.uint8)

def load_frame_data(prefix, frame_ix, camera_names=['top','bottom'], frame_size=(192,192)):
    
    matched_frames = {camera_name:ix for camera_name,ix in zip(
        camera_names,load_matched_frames(prefix, camera_names)[frame_ix])}
    
    frame_data = {}
    for name,suffixes in {
        'masks':     ['ir','depth','mouse_mask','occl_mask'],
        'ortho':     ['ir_ortho','depth_ortho','occl_ortho','missing_ortho'],
        'inpainted': ['ir_inpainted','depth_inpainted'],
        'aligned':   ['ir_aligned','depth_aligned'],
        'encoded':   ['ir_encoded','depth_encoded']}.items():
        
        data = {}
        for camera_name,suffix in product(camera_names,suffixes):
            
            file_path = prefix+'.'+camera_name+'.'+suffix+'.avi'
            if os.path.exists(file_path) and count_frames(file_path)>frame_ix:
                
                if suffix in ['ir','depth']: data[(camera_name,suffix)] = read_frames(
                    prefix+'.'+camera_name+'.'+suffix+'.avi', 
                    [matched_frames[camera_name]], pixel_format='gray16').squeeze()
                    
                elif 'mask' in suffix: data[(camera_name,suffix)] = read_frames(
                    prefix+'.'+camera_name+'.'+suffix+'.avi', 
                    [frame_ix], pixel_format='gray8').squeeze()
                    
                else: data[(camera_name,suffix)] = read_frames(
                    prefix+'.'+camera_name+'.'+suffix+'.avi', 
                    [frame_ix], frame_size=frame_size, pixel_format='gray8').squeeze()
                    
        frame_data[name] = data
            
    return frame_data
    
def to_gray(x):
    return np.repeat(x[:,:,None],3,axis=2)

def to_color(x, channel):
    return np.repeat(x[:,:,None],3,axis=2)*np.eye(3)[channel]

def format_mask_panels(masks, camera_name, frame_size=(192,192), 
                       alpha=0.3, depth_min=550, depth_gain=4):
    
    crop_size = frame_size[0]//2
    if masks[(camera_name,'mouse_mask')].max()==0: centroid=np.array(frame_size)
    else: centroid = np.mean(masks[(camera_name,'mouse_mask')].nonzero(),axis=1).astype(int)
    ir         = crop(masks[(camera_name,'ir')],         centroid, crop_size).astype(float)
    depth      = crop(masks[(camera_name,'depth')],      centroid, crop_size).astype(float)
    mouse_mask = crop(masks[(camera_name,'mouse_mask')], centroid, crop_size).astype(float)*255
    occl_mask  = crop(masks[(camera_name,'occl_mask')],  centroid, crop_size).astype(float)*255

    ir = rescale_ir(ir)
    depth = np.clip((depth-depth_min)*depth_gain, 0, 255)

    return [
        (to_gray(ir)    * (1-alpha) + to_color(mouse_mask,0) * alpha).astype(np.uint8),
        (to_gray(depth) * (1-alpha) + to_color(mouse_mask,0) * alpha).astype(np.uint8),
        (to_gray(ir)    * (1-alpha) + to_color(occl_mask,1)  * alpha).astype(np.uint8),
        (to_gray(depth) * (1-alpha) + to_color(occl_mask,1)  * alpha).astype(np.uint8)]

 
def add_titles(panels, titles, text_origin=(5,18), 
               text_size=0.4, text_color=(255,255,255)):
    
    return [cv2.putText(panel.copy(), title, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 
                        text_size, text_color, 1, cv2.LINE_AA) 
            for panel,title in zip(panels,titles)]


def get_qc_frame(*, masks, ortho, inpainted, aligned, encoded, 
                 frame_size=(192,192), camera_names=['top','bottom'], 
                 depth_mins=[520,600], depth_gain=4, mode='columns', 
                 encoded_ir_clip=(90,215),encoded_depth_clip=(115,180),
                 mask_options={}, text_options={}):
    
    all_panels = []
    
    if len(masks)>0:
        titles = ['mouse.ir','mouse.depth','occlusion.ir','occlusion.depth']
        for camera_name,depth_min in zip(camera_names,depth_mins):
            panels = format_mask_panels(masks, camera_name, frame_size=frame_size, depth_min=depth_min, **mask_options)
            all_panels.append(add_titles(panels, [t+'.'+camera_name for t in titles], **text_options))
            
    if len(ortho)>0:
        panels,titles = [],[]
        for camera_name in camera_names:
            missing = ortho[(camera_name,'missing_ortho')].astype(float)*255
            occl    = ortho[(camera_name,'occl_ortho')].astype(float)*255
            ir      = np.where((missing+occl)==0, ortho[(camera_name,'ir_ortho')], 0).astype(float)
            depth   = np.where((missing+occl)==0, ortho[(camera_name,'depth_ortho')], 0).astype(float)
            depth   = np.clip(depth*depth_gain, 0, 255)
            panels.append(to_gray(ir)+to_color(missing,2)+to_color(occl,1).astype(np.uint8))
            panels.append((to_gray(depth)+to_color(missing,2)+to_color(occl,1)).astype(np.uint8))
            titles += ['ortho.ir.'+camera_name, 'ortho.depth.'+camera_name]
        all_panels.append(add_titles(panels, titles, **text_options))
            
    if len(inpainted)>0:
        panels,titles = [],[]
        for camera_name in camera_names:
            ir    = inpainted[(camera_name,'ir_inpainted')]
            depth = inpainted[(camera_name,'depth_inpainted')].astype(float)
            depth = np.clip(depth*depth_gain, 0, 255)
            panels += [to_gray(ir).astype(np.uint8), to_gray(depth).astype(np.uint8)]
            titles += ['inpainted.ir.'+camera_name, 'inpainted.depth.'+camera_name]
        all_panels.append(add_titles(panels, titles, **text_options))
          
    if len(aligned)>0:
        panels,titles = [],[]
        for camera_name in camera_names:
            ir    = clip_and_color(aligned[(camera_name,'ir_aligned')],    *encoded_ir_clip)
            depth = clip_and_color(aligned[(camera_name,'depth_aligned')], *encoded_depth_clip)
            panels += [ir, depth]
            titles += ['aligned.ir.'+camera_name, 'aligned.depth.'+camera_name]
        all_panels.append(add_titles(panels, titles, **text_options))
        
    if len(encoded)>0:
        panels,titles = [],[]
        for camera_name in camera_names:
            ir    = clip_and_color(encoded[(camera_name,'ir_encoded')],    *encoded_ir_clip)
            depth = clip_and_color(encoded[(camera_name,'depth_encoded')], *encoded_depth_clip)
            panels += [ir, depth]
            titles += ['encoded.ir.'+camera_name, 'encoded.depth.'+camera_name]
        all_panels.append(add_titles(panels, titles, **text_options))

    if mode=='rows': return np.vstack([np.hstack(panels) for panels in all_panels]).astype(np.uint8)
    else: return np.hstack([np.vstack(panels) for panels in all_panels]).astype(np.uint8)
    
def save_qc_movie(prefix, num_frames=None, **kwargs):
    camera_names = ['top','bottom']
    ortho_readers, inpainted_readers, aligned_readers, encoded_readers = {},{},{},{}
    mask_readers = {(camera_name,k): imageio.get_reader(
        prefix+'.'+camera_name+'.'+k+'.avi'
    ) for camera_name,k in product(camera_names, ['mouse_mask','occl_mask'])}
    
    if os.path.exists(prefix+'.'+camera_names[0]+'.ir_ortho.avi'):
        ortho_readers = {(camera_name,k): imageio.get_reader(
            prefix+'.'+camera_name+'.'+k+'.avi'
        ) for camera_name,k in product(camera_names, ['ir_ortho','depth_ortho','missing_ortho','occl_ortho'])}
    
    if os.path.exists(prefix+'.'+camera_names[0]+'.ir_inpainted.avi'):
        inapinted_readers = {(camera_name,k): imageio.get_reader(
            prefix+'.'+camera_name+'.'+k+'.avi'
        ) for camera_name,k in product(camera_names, ['ir_inpainted','depth_inpainted'])}
    
    if os.path.exists(prefix+'.'+camera_names[0]+'.ir_aligned.avi'):
        aligned_readers = {(camera_name,k): imageio.get_reader(
            prefix+'.'+camera_name+'.'+k+'.avi'
        ) for camera_name,k in product(camera_names, ['ir_aligned','depth_aligned'])}
    
    if os.path.exists(prefix+'.'+camera_names[0]+'.ir_encoded.avi'):
        encoded_readers = {(camera_name,k): imageio.get_reader(
            prefix+'.'+camera_name+'.'+k+'.avi'
        ) for camera_name,k in product(camera_names, ['ir_encoded','depth_encoded'])}
    
    matched_frames = load_matched_frames(prefix, ['top','bottom'])
    with videoReader(prefix+'.top.ir.avi',       matched_frames[:,0]) as top_ir_reader, \
         videoReader(prefix+'.top.depth.avi',    matched_frames[:,0]) as top_depth_reader, \
         videoReader(prefix+'.bottom.ir.avi',    matched_frames[:,1]) as bottom_ir_reader, \
         videoReader(prefix+'.bottom.depth.avi', matched_frames[:,1]) as bottom_depth_reader, \
         imageio.get_writer(prefix+'.QC.mp4', pixelformat='yuv420p', fps=30, quality=6) as writer:
        
        for ix,(ir_top, depth_top, ir_bottom, depth_bottom) in tqdm.tqdm(enumerate(zip(
            top_ir_reader, top_depth_reader, bottom_ir_reader, bottom_depth_reader))):
            
            frame_data = {
                'masks':     {k:reader.get_data(ix)[:,:,0] for k,reader in mask_readers.items()},
                'ortho':     {k:reader.get_data(ix)[:,:,0] for k,reader in ortho_readers.items()},
                'inpainted': {k:reader.get_data(ix)[:,:,0] for k,reader in inpainted_readers.items()},
                'aligned':   {k:reader.get_data(ix)[:,:,0] for k,reader in aligned_readers.items()},
                'encoded':   {k:reader.get_data(ix)[:,:,0] for k,reader in encoded_readers.items()}}
            
            frame_data['masks'][('top','ir')] = ir_top
            frame_data['masks'][('top','depth')] = depth_top
            frame_data['masks'][('bottom','ir')] = ir_bottom
            frame_data['masks'][('bottom','depth')] = depth_bottom
            masks = {k:reader.get_data(ix) for k,reader in mask_readers.items()}
            
            frame = get_qc_frame(**frame_data, **kwargs, camera_names=camera_names)
            frame = cv2.putText(frame, repr(ix),(10,frame.shape[0]-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),1,cv2.LINE_AA)
            writer.append_data(frame)
            if num_frames is not None and ix>=num_frames: break
    print('QC video written to '+prefix+'.QC.mp4')
                
def grab_qc_frame(prefix, frame_ix, **kwargs):
    frame_data = load_frame_data(prefix, frame_ix)
    return get_qc_frame(**frame_data, **kwargs)
