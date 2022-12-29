import torch, kornia
import numpy as np
import tqdm
from itertools import product
from top_bottom_moseq.io import videoWriter, read_frames
from top_bottom_moseq.util import check_if_already_done

def vec_to_angle(v, degrees=False):
    a = torch.arctan(v[:,1]/v[:,0]) + np.pi*(v[:,0]>0)
    if degrees: a = a / np.pi * 180
    return a

def angle_to_vec(a, degrees=False):
    if degrees: a = a / 180 * np.pi 
    return -torch.hstack([torch.cos(a)[:,None],torch.sin(a)[:,None]])

    
class LocalizationNet(torch.nn.Module):   
    def __init__(self, input_shape, 
                 layer_sizes=[32,32,64,128], 
                 hidden_units=128, 
                 max_translation=64, 
                 max_depth_shift=10):
        
        super().__init__()
        self.multiplier = torch.tensor([1,1,max_translation,max_translation,max_depth_shift])[None].to('cuda')
        self.down_blocks = torch.nn.ModuleList()
        for in_c,out_c in zip([input_shape[0]]+layer_sizes[:-1],layer_sizes):
            self.down_blocks.append(self.down_block(in_c,out_c))
            
        final_layer_dims = [int(d/(2**len(layer_sizes))-2*len(layer_sizes)+2) for d in input_shape[1:]]
        self.regression_block = torch.nn.Sequential(
            torch.nn.Linear(final_layer_dims[0]*final_layer_dims[1]*layer_sizes[-1], hidden_units),
            torch.nn.LeakyReLU(), 
            torch.nn.Linear(hidden_units, 5),
            torch.nn.Tanh())

    @staticmethod
    def down_block(in_channels, out_channels):
        return torch.nn.Sequential(
        torch.nn.Conv2d(in_channels, out_channels, kernel_size=(3,3)),
        torch.nn.BatchNorm2d(out_channels), torch.nn.LeakyReLU(),
        torch.nn.Conv2d(out_channels, out_channels, kernel_size=(3,3), dilation=(2,2)),
        torch.nn.BatchNorm2d(out_channels), torch.nn.LeakyReLU(),
        torch.nn.MaxPool2d(kernel_size=(2,2), stride=(2,2)))

    def forward(self, x):
        for block in self.down_blocks: x = block(x)
        theta = self.regression_block(x.reshape(x.shape[0],-1))
        return theta * self.multiplier
    
    
class Autoencoder(torch.nn.Module):
    def __init__(self, input_shape, conv_features=[16,32,32], dense_features=[10]):
        super(Autoencoder, self).__init__()
        self.first_dense_dim = int(input_shape[1]*input_shape[2]*conv_features[-1]/4**len(conv_features))
        self.last_conv_dim = (conv_features[-1],int(input_shape[1]/2**len(conv_features)),int(input_shape[2]/2**len(conv_features)))
        conv_dims = [input_shape[0]]+conv_features
        dense_dims = [self.first_dense_dim]+dense_features
             
        self.conv_encoders = torch.nn.ModuleList([])
        for d_in,d_out in zip(conv_dims[:-1],conv_dims[1:]):
            self.conv_encoders.append(torch.nn.Conv2d(d_in, d_out, (3,3), padding=(1,1)))
            self.conv_encoders.append(torch.nn.BatchNorm2d(d_out))
            self.conv_encoders.append(torch.nn.Tanh())
            self.conv_encoders.append(torch.nn.Conv2d(d_out, d_out, (3,3), stride=(2,2), padding=(1,1)))
            self.conv_encoders.append(torch.nn.BatchNorm2d(d_out))
            self.conv_encoders.append(torch.nn.Tanh())

        self.dense_encoders = torch.nn.ModuleList([])
        for d_in,d_out in zip(dense_dims[:-1],dense_dims[1:]):
            self.dense_encoders.append(torch.nn.Linear(d_in, d_out))
                
        self.dense_decoders = torch.nn.ModuleList([])
        for d_in,d_out in zip(dense_dims[::-1][:-1],dense_dims[::-1][1:]):
            self.dense_decoders.append(torch.nn.Linear(d_in, d_out))
       
        self.conv_decoders = torch.nn.ModuleList([])
        for d_in,d_out in zip(conv_dims[::-1][:-1],conv_dims[::-1][1:]):
            self.conv_decoders.append(torch.nn.UpsamplingBilinear2d(scale_factor=2))
            self.conv_decoders.append(torch.nn.Conv2d(d_in, d_out, (3,3), padding=(1,1)))
            self.conv_decoders.append(torch.nn.BatchNorm2d(d_out))
            self.conv_decoders.append(torch.nn.Tanh()) 
            self.conv_decoders.append(torch.nn.Conv2d(d_out, d_out, (3,3), padding=(1,1)))
            self.conv_decoders.append(torch.nn.BatchNorm2d(d_out))
            self.conv_decoders.append(torch.nn.Tanh()) 
                

    def encode(self,x):
        for layer in self.conv_encoders: x = layer(x)
        x = x.view(-1,self.first_dense_dim)
        for layer in self.dense_encoders: x = layer(x)
        return x

    def decode(self,x):
        for layer in self.dense_decoders: x = layer(x)
        x = x.view(-1,*self.last_conv_dim)
        for layer in self.conv_decoders: x = layer(x)
        return x
        
    def forward(self, x):
        x = self.encode(x)
        return self.decode(x)



class Transformer(torch.nn.Module):
    def __init__(self, image_size=(192,192)):
        super().__init__()
        self.image_size = image_size
        self.depth_channels = torch.tensor([0,2])
        self.ir_channels = torch.tensor([1,3])
        self.ir_shift = -0.3
        self.center = torch.tensor([image_size[0]/2,image_size[1]/2]).unsqueeze(0).to('cuda')
    
    def forward(self, x, m, theta, invert=False):
        angle = vec_to_angle(theta[:,:2], degrees=True)
        center = self.center.expand((x.shape[0],2))
        
        if invert:
            x = kornia.geometry.transform.rotate(x,-angle,center)
            x = kornia.geometry.transform.translate(x,-theta[:,2:4])
            m = kornia.geometry.transform.rotate(m,-angle,center)
            m = kornia.geometry.transform.translate(m,-theta[:,2:4])
            for i in self.depth_channels: x[:,i,:,:] += m[:,i,:,:]*(-theta[:,4,None,None])
            for i in self.ir_channels: x[:,i,:,:] += m[:,i,:,:]*(-self.ir_shift)

        else:
            x = kornia.geometry.transform.translate(x,theta[:,2:4])
            x = kornia.geometry.transform.rotate(x,angle,center)
            m = kornia.geometry.transform.translate(m,theta[:,2:4])
            m = kornia.geometry.transform.rotate(m,angle,center)
            for i in self.depth_channels: x[:,i,:,:] += m[:,i,:,:]*(theta[:,4,None,None])
            for i in self.ir_channels: x[:,i,:,:] += m[:,i,:,:]*self.ir_shift
        return x,m
        
def load_inpainting_videos(prefix, length, camera_names, channels, frame_size):
    X = np.empty((length, len(camera_names)*len(channels), *frame_size), dtype=np.uint8)
    for i,(camera,channel) in tqdm.tqdm(
        enumerate(product(camera_names,channels)),
        desc='Loading inpainted videos'):
        file_path = prefix+'.'+camera+'.'+channel+'_inpainted.avi'
        X[:,i] = read_frames(file_path, frame_size=frame_size, frames=range(length))
    return X

def load_models(autoencoder_weights, localization_weights, frame_size):
    autoencoder = Autoencoder((4,*frame_size)).to('cuda').eval()
    localizationNet = LocalizationNet((4,*frame_size)).to('cuda').eval()
    autoencoder.load_state_dict(torch.load(autoencoder_weights))
    localizationNet.load_state_dict(torch.load(localization_weights))
    return autoencoder,localizationNet,Transformer()

def encode_session(prefix, autoencoder_weights, localization_weights, 
                   frame_size=(192,192), channels=['depth','ir'], 
                   camera_names=['top','bottom'],
                   overwrite=False):

    length = np.load(prefix+'.crop_centers.npy').shape[0]

    # Don't process if already done!
    out_file_list = [prefix + f'.{cam}.{movie_type}.avi' for movie_type in ['ir_aligned', 'depth_aligned', 'ir_encoded', 'depth_encoded'] for cam in ['top', 'bottom']]
    if (not overwrite) and all([check_if_already_done(file, length, overwrite=overwrite) for file in out_file_list]):
        print('Movies already encoded, continuing...')
        return

    X = load_inpainting_videos(prefix, length, camera_names, channels, frame_size)
    
    autoencoder,localizationNet,transformer = load_models(
        autoencoder_weights, localization_weights, frame_size)
        
    latents,thetas = [],[]
    
    with videoWriter(prefix+'.top.ir_aligned.avi')       as top_ir_aligned_writer, \
         videoWriter(prefix+'.top.depth_aligned.avi')    as top_depth_aligned_writer, \
         videoWriter(prefix+'.top.ir_encoded.avi')       as top_ir_encoded_writer, \
         videoWriter(prefix+'.top.depth_encoded.avi')    as top_depth_encoded_writer, \
         videoWriter(prefix+'.bottom.ir_aligned.avi')    as bottom_ir_aligned_writer, \
         videoWriter(prefix+'.bottom.depth_aligned.avi') as bottom_depth_aligned_writer, \
         videoWriter(prefix+'.bottom.ir_encoded.avi')    as bottom_ir_encoded_writer, \
         videoWriter(prefix+'.bottom.depth_encoded.avi') as bottom_depth_encoded_writer, \
         torch.no_grad():
        
        for ix in tqdm.trange(0,length, desc='alignment and encoding'):
            XX = torch.tensor(X[ix][None].astype(np.float32)/255).to('cuda')
            theta = localizationNet(XX)
            XX_a = transformer(XX, (XX>0.05).float(), theta)[0]
            XX_e = autoencoder(XX_a)
            X_aligned = (np.clip(XX_a.detach().cpu().numpy()+.5,0,1)*255).astype(np.uint8)
            X_encoded = (np.clip(XX_e.detach().cpu().numpy()+.5,0,1)*255).astype(np.uint8)
            latents.append(autoencoder.encode(XX_a).detach().cpu().numpy())
            thetas.append(theta.detach().cpu().numpy())

            top_depth_aligned_writer.append(X_aligned[:,0])
            top_ir_aligned_writer.append(X_aligned[:,1])
            bottom_depth_aligned_writer.append(X_aligned[:,2])
            bottom_ir_aligned_writer.append(X_aligned[:,3])
            top_depth_encoded_writer.append(X_encoded[:,0])
            top_ir_encoded_writer.append(X_encoded[:,1])
            bottom_depth_encoded_writer.append(X_encoded[:,2])
            bottom_ir_encoded_writer.append(X_encoded[:,3])

    np.save(prefix+'.latents.npy', latents)
    np.save(prefix+'.thetas.npy', thetas)
    