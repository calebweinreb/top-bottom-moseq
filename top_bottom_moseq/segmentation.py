import numpy as np
import torch
import cv2
import tqdm
import segmentation_models_pytorch
from top_bottom_moseq.util import rescale_ir, load_matched_frames, crop, uncrop
from top_bottom_moseq.io import videoReader, videoWriter

def load_segmentation_model(weights_path):
    model = segmentation_models_pytorch.UnetPlusPlus(
        in_channels=1, encoder_weights=None,activation='sigmoid', encoder_depth=5,
        decoder_channels = (128, 128, 64, 32, 16),encoder_name='efficientnet-b0').to('cuda')
    model.load_state_dict(torch.load(weights_path))
    return model.eval()

def get_crop_center(mask, crop_size):
    crop_center = np.array(mask.nonzero()).mean(1)
    return np.minimum(np.maximum(crop_center, crop_size), np.array(mask.shape)-crop_size)


def to_numpy(X):
    return X.detach().cpu().numpy()

def remove_small_components(mask, min_size):
    removed = np.zeros_like(mask)
    ccs = cv2.connectedComponents(mask.astype(np.uint8))[1]
    for i in range(1,ccs.max()+1):
        if (ccs==i).sum()<min_size:
            removed = np.where(ccs==i, 1, removed)
    return np.where(removed, 0, mask), removed

def segment_session(prefix, 
                    mouse_model_weights, 
                    occlusion_model_weights, 
                    camera_names=['top','bottom'], 
                    min_component_size=500, 
                    crop_size=96, 
                    threshold=0.5
                   ):
    mouse_model = load_segmentation_model(mouse_model_weights)
    occl_model = load_segmentation_model(occlusion_model_weights)
    matched_frames = load_matched_frames(prefix, camera_names)
    
    for camera,frames in zip(camera_names,matched_frames.T):
        
        with videoReader(prefix+'.{}.ir.avi'.format(camera), frames) as ir_reader, \
             videoWriter(prefix+'.{}.mouse_mask.avi'.format(camera)) as mouse_writer, \
             videoWriter(prefix+'.{}.occl_mask.avi'.format(camera))  as occl_writer, \
             torch.no_grad():

            for ir in tqdm.tqdm(ir_reader, desc='segmentation, '+camera):
                ir = rescale_ir(ir)[None,None,:,:]/255
                X = torch.from_numpy(ir.astype(np.float32)).to('cuda')
                mouse_mask = to_numpy(mouse_model(X).squeeze())>threshold
                occl_mask = np.zeros_like(mouse_mask)

                if mouse_mask.sum() > 0:
                    crop_center = get_crop_center(mouse_mask, crop_size=crop_size)
                    occl_mask_cropped = to_numpy(occl_model(crop(X, crop_center, crop_size))).squeeze()>0.5
                    occl_mask = uncrop(occl_mask_cropped, crop_center, crop_size, mouse_mask.shape)

                mouse_mask, removed = remove_small_components(mouse_mask, min_component_size)
                occl_mask = np.any([occl_mask, removed], axis=0)
                
                mouse_writer.append(mouse_mask)
                occl_writer.append(occl_mask)

