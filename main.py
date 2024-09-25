#%%
from utils import *
#%%
delete_all_files('output')
#%%
delete_all_files('frames')
# %%
frames_directory = 'frames'
output_filename = 'output\output_video.mp4' 
#%%
create_video_from_frames(frames_directory, output_filename)
#%%
create_video_from_frames_general(frames_directory, output_filename)
# %%
