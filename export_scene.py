import moviepy.video.io.ImageSequenceClip
from glob import glob


def save_video(img_folder, fps=30):
    image_files = sorted(glob(f'{img_folder}/*.png'))
    clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
    clip.write_videofile(f'{img_folder}.mp4')

