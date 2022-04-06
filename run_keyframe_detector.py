from lib import key_frame_detector
import sys
import os

if __name__ == "__main__":
	input_dir = sys.argv[1]
	output_dir = sys.argv[2]
	for dirname in os.listdir(input_dir):
		cur_path = os.path.join(input_dir, dirname)
		for filename in os.listdir(cur_path):
			if filename.endswith("mp4"):
				video_path = os.path.join(cur_path, filename)
				print(filename)
			if filename.endswith("srt"):
				subfile_path = os.path.join(cur_path, filename)
		output_path = os.path.join(sys.argv[2], dirname)
		detector = key_frame_detector.KeyFrameDetector(video_path, subfile_path)
		detector.detect(output_path, frame_rate=10, K=5)