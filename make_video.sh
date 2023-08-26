ffmpeg -framerate 30 -i results/%05d.png -c:v libx264 -pix_fmt yuv420p output.mp4
