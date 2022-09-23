import numpy as np
import av
import subprocess
import datetime
        

def count_frames(file_name):
    with av.open(file_name, 'r') as reader:
        return reader.streams.video[0].frames          

class videoWriter():
    def __init__(self, file_name, **ffmpeg_options):
        self.pipe = None
        self.file_name = file_name
        self.ffmpeg_options = ffmpeg_options
        
    def __enter__(self):
        return self
    
    def __exit__(self, type, value, traceback):
        if self.pipe is not None:
            self.pipe.stdin.close()
        
    def append(self, frames):
        if len(frames.shape)==2: frames = frames[None]
        self.pipe = write_frames(self.file_name, frames, pipe=self.pipe, **self.ffmpeg_options)

class videoReader():
    def __init__(self, file_name, frame_ixs=None):
        self.file_name = file_name
        self.frame_ixs = frame_ixs

    def __enter__(self):
        self.reader = av.open(self.file_name, 'r')
        self.reader.streams.video[0].thread_type = "AUTO"
        
        frame_mask = np.zeros(self.reader.streams.video[0].frames)
        if self.frame_ixs is not None:
            assert self.frame_ixs.max() < len(frame_mask), 'frame_ixs exceeds the video length'
            frame_mask[self.frame_ixs.astype(int)] = 1
        else: frame_mask[:] = 1
            
        return self.frame_gen(self.reader.decode(video=0), frame_mask)

    def frame_gen(self, reader, frame_mask):
        for ix,frame in enumerate(reader):
            if frame_mask[ix]: yield frame.to_ndarray()

    def __exit__(self, exc_type,exc_value, exc_traceback):
        self.reader.close()            
                
def write_frames(filename, frames, 
                 threads=6, fps=30, crf=10,
                 pixel_format='gray8', codec='ffv1',
                 pipe=None, slices=24, slicecrc=1):
    
    frame_size = '{0:d}x{1:d}'.format(frames.shape[2], frames.shape[1])
    command = ['ffmpeg',
               '-y',
               '-loglevel', 'fatal',
               '-framerate', str(fps),
               '-f', 'rawvideo',
               '-s', frame_size,
               '-pix_fmt', pixel_format,
               '-i', '-',
               '-an',
               '-crf',str(crf),
               '-vcodec', codec,
               '-preset', 'ultrafast',
               '-threads', str(threads),
               '-slices', str(slices),
               '-slicecrc', str(slicecrc),
               '-r', str(fps),
               filename]

    if not pipe: pipe = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    dtype = np.uint16 if pixel_format.startswith('gray16') else np.uint8
    for i in range(frames.shape[0]): pipe.stdin.write(frames[i,:,:].astype(dtype).tobytes())
    return pipe


def read_frames(filename, frames, threads=6, fps=30, frames_is_timestamp=False,
                pixel_format='gray16', frame_size=(640,576),
                slices=24, slicecrc=1, get_cmd=False):
    """
    Reads in frames from the .mp4/.avi file using a pipe from ffmpeg.
    Args:
        filename (str): filename to get frames from
        frames (list or 1d numpy array): list of frames to grab
        threads (int): number of threads to use for decode
        fps (int): frame rate of camera in Hz
        pixel_format (str): ffmpeg pixel format of data
        frame_size (str): wxh frame size in pixels
        slices (int): number of slices to use for decode
        slicecrc (int): check integrity of slices
    Returns:
        3d numpy array:  frames x h x w
    """

    if frames_is_timestamp: start_time = str(datetime.timedelta(seconds=frames[0]))
    else: start_time = str(datetime.timedelta(seconds=frames[0]/fps))
    
    command = [
        'ffmpeg',
        '-loglevel', 'fatal',
        '-vsync','0',
        '-ss', start_time,
        '-i', filename,
        '-vframes', str(len(frames)),
        '-f', 'image2pipe',
        '-s', '{:d}x{:d}'.format(frame_size[0], frame_size[1]),
        '-pix_fmt', pixel_format,
        '-threads', str(threads),
        '-slices', str(slices),
        '-slicecrc', str(slicecrc),
        '-vcodec', 'rawvideo',
        '-'
    ]

    if get_cmd:
        return command
    
  
    pipe = subprocess.Popen(command, stderr=subprocess.PIPE, stdout=subprocess.PIPE)
    out, err = pipe.communicate()
    if(err):
        print('error', err)
        return None
    
    dtype = ('uint16' if '16' in pixel_format else 'uint8')
    video = np.frombuffer(out, dtype=dtype).reshape((len(frames), frame_size[1], frame_size[0]))
    return video
