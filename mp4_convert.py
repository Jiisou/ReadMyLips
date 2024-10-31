from moviepy.editor import VideoFileClip

def convert_to_mp4(input_path, output_path):
    clip = VideoFileClip(input_path)
    clip.write_videofile(output_path, codec="libx264")

convert_to_mp4("세바시_고속노화.mov", "output_video.mp4")
