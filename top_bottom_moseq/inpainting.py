
from itertools import product
import numpy as np
import tqdm
import torch
import torch.nn as nn
from kornia.morphology import dilation
from top_bottom_moseq.io import read_frames, videoWriter
from top_bottom_moseq.util import check_if_already_done

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = torch.nn.functional.pad(
            x1, [diffX // 2, diffX - diffX // 2,
            diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, down_layers=[64,64,128,128,256], up_layers=[128,128,64,64]):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, down_layers[0])
        self.down1 = Down(down_layers[0], down_layers[1])
        self.down2 = Down(down_layers[1], down_layers[2])
        self.down3 = Down(down_layers[2], down_layers[3])
        self.down4 = Down(down_layers[3], down_layers[4])
        self.up1 = Up(down_layers[3]+down_layers[4], up_layers[0])
        self.up2 = Up(down_layers[2]+up_layers[0], up_layers[1])
        self.up3 = Up(down_layers[1]+up_layers[1], up_layers[2])
        self.up4 = Up(down_layers[0]+up_layers[2], up_layers[3])
        self.outc = OutConv(up_layers[3], n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        return torch.sigmoid(out)
    
    
    
def load_inpainting_models(inpainting_weights):
    init_net = UNet(6,8, down_layers=[64, 64, 64, 128, 128], up_layers=[64, 64, 64, 64]).to('cuda').eval()
    step_net = UNet(14,8, down_layers=[64, 64, 64, 128, 128], up_layers=[64, 64, 64, 64]).to('cuda').eval()
    comb_net = UNet(22,8, down_layers=[96,96,96,128,128], up_layers=[64, 64, 64, 64]).to('cuda').eval()
    init_net.load_state_dict(torch.load(inpainting_weights.format('init_net')))
    step_net.load_state_dict(torch.load(inpainting_weights.format('step_net')))
    comb_net.load_state_dict(torch.load(inpainting_weights.format('comb_net')))
    return init_net,step_net,comb_net
 
    
def get_strel(size):
    X,Y = torch.meshgrid(torch.arange(size),torch.arange(size), indexing='ij')
    return (((X-size//2)**2 + (Y-size//2)**2) <= (size-1)).float()

def dilate(x, strel, iters):
    for _ in range(iters):
        x = dilation(x,strel)
    return x

def mask_dilation(x, cuda=True):
    with torch.no_grad():
        strel = get_strel(5).to('cuda')
        occl_top = torch.clip((dilate(x[:,2].unsqueeze(1),strel,1).squeeze(1) + x[:,3]),0,1)
        occl_bot = torch.clip((dilate(x[:,6].unsqueeze(1),strel,1).squeeze(1) + x[:,7]),0,1)
        return torch.stack([
            x[:,0] * (1-occl_top),
            x[:,1] * (1-occl_top),
            x[:,4] * (1-occl_bot),
            x[:,5] * (1-occl_bot),
            occl_top, occl_bot], dim=1)

def load_ortho_videos(prefix, length, camera_names, channels, scale_factors, frame_size):
    X = np.empty((length, len(camera_names)*len(channels), *frame_size), dtype=np.uint8)
    for i,(camera,(channel,scale)) in tqdm.tqdm(
        enumerate(product(camera_names,(zip(channels,scale_factors)))),
        desc='Loading orthographic projections'):
        file_path = prefix+'.'+camera+'.'+channel+'_ortho.avi'
        X[:,i] = read_frames(file_path, frame_size=frame_size, frames=range(length))*scale  
    return X


def unpack(x):
    x = x.astype(np.float32) / 255
    return torch.tensor(x)[None].to('cuda')

def pack(y):
    y = y.clone().detach().cpu().numpy().squeeze()
    return np.clip(y*255,0,255).astype(np.uint8)

def inpaint_session(prefix, inpainting_weights, lag=2,
                    channels=['depth','ir','occl','missing'],
                    camera_names=['top','bottom'], 
                    scale_factors=[1,1,255,255],
                    frame_size=(192,192),
                    overwrite=False):
    
    init_net, step_net, comb_net = load_inpainting_models(inpainting_weights)
    length = np.load(prefix+'.crop_centers.npy').shape[0]
    
    # Don't process if already done!
    out_file_list = [prefix + f'.{cam}.{movie_type}.avi' for movie_type in ['ir_inpainted', 'depth_inpainted'] for cam in ['top', 'bottom']]
    if (not overwrite) and all([check_if_already_done(file, length, overwrite=overwrite) for file in out_file_list]):
        print('Movies already in-painted, continuing...')
        return

    # load full recording
    X = load_ortho_videos(prefix, length, camera_names, channels, scale_factors, frame_size)
    for ix in tqdm.trange(1,length-1, desc='spreading mask'):
        X[ix,2] = np.any([X[ix,2],np.all([X[ix,0]>0,np.any(X[ix-lag:ix+lag+1,0]==0,axis=0)],axis=0)],axis=0).astype(float)*255
        X[ix,6] = np.any([X[ix,6],np.all([X[ix,4]>0,np.any(X[ix-lag:ix+lag+1,4]==0,axis=0)],axis=0)],axis=0).astype(float)*255

    # rev pass            
    Y_rev = np.zeros((length,8,192,192),dtype=np.uint8)
    with torch.no_grad():
        y = init_net(mask_dilation(unpack(X[-1])))
        for ix in tqdm.tqdm(list(range(0,length))[::-1], desc='reverse pass'):
            y = step_net(torch.cat([mask_dilation(unpack(X[ix])),y],axis=1))
            Y_rev[ix] = pack(y)                

    # forward pass
    with videoWriter(prefix+'.top.ir_inpainted.avi')       as top_ir_writer, \
         videoWriter(prefix+'.top.depth_inpainted.avi')    as top_depth_writer, \
         videoWriter(prefix+'.bottom.ir_inpainted.avi')    as bottom_ir_writer, \
         videoWriter(prefix+'.bottom.depth_inpainted.avi') as bottom_depth_writer, \
         torch.no_grad():

        y_fwd = init_net(mask_dilation(unpack(X[0])))
        for ix in tqdm.trange(0,length, desc='forward/combined pass'):
            x = mask_dilation(unpack(X[ix]))
            y_fwd = step_net(torch.cat([x,y_fwd],axis=1))
            y_rev = unpack(Y_rev[ix])
            y = pack(comb_net(torch.cat([x,y_fwd,y_rev],axis=1)))

            top_depth_writer.append(y[0])
            top_ir_writer.append(y[1])
            bottom_depth_writer.append(y[2])
            bottom_ir_writer.append(y[3])


