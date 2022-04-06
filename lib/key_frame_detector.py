import cv2
import pysubs2
import argparse
import time
import re
from nltk.corpus import stopwords
from string import punctuation
import os
import pickle
import matplotlib.pyplot as plt
import shutil

class KeyFrameDetector(object):
    def __init__(self, video_path, subfile_path):
        self.video = cv2.VideoCapture(video_path)
        if not self.video.isOpened():
            print("ERROR: Cannot open video {}".format(video_path))
        self.FPS = self.video.get(cv2.CAP_PROP_FPS)
        try:
            self.sub = pysubs2.load(subfile_path)
        except:
            print("ERROR: No fuch subtitle file {}".format(subfile_path))
        # remove stop words and punctuation
        self.remove = remove = set(stopwords.words('english') + list(punctuation) + [""])
        self.sub_range = self.calc_sub_range()
        self.sub_keywords = self.extract_sub_keywords()
        self.key_frames = []

    def calc_sub_range(self):
        res = []
        # calculate frame# by ms
        for s in self.sub:
            res.append((int(self.FPS*s.start/1e3), int(self.FPS*s.end/1e3)))
        return res

    def plaintext2keywords(self, text):
        # split plaintext and remove the stopwords
        # remove the number
        text = re.sub(r'\d+', '', text)
        kw_list = [w for w in re.split(r'\W+', text.lower()) if w not in self.remove]
        # remove the same words
        return list(set(kw_list))

    def extract_sub_keywords(self):
        res = []
        for s in self.sub:
            keywords_list = self.plaintext2keywords(s.plaintext)
            res.append(keywords_list)
        return res

    def detect(self, output_dir, threshold=0.8, frame_rate=10, min_inter=100, K=5):
        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)
            #print("Mkdir:{}".format(output_dir))
        img_output_dir = os.path.join(output_dir, "key_frame_image")
        os.mkdir(img_output_dir)
        #print("Mkdir:{}".format(img_output_dir))
        # init
        shot_s = 0
        shot_e = -1
        in_shot = True
        f_count = 0
        shot_count = 0
        start = time.time()
        _, f0 = self.video.read()
        # do shot segmentation based on grayscale color histogram
        h0 = cv2.calcHist([cv2.cvtColor(f0, cv2.COLOR_BGR2GRAY)], [0], None, [256], [0,256])
        shot_list = []
        # keyframe buffer
        frame_buf = []
        cur_sub_idx = 0
        cur_sub = self.sub_range[cur_sub_idx]
        # result list
        res_key_frames = []
        fnum2kwlist = {}

        insub_skip = False
        outsub_skip = False

        while True:
            ret, f1 = self.video.read()
            if not ret:
                break
            f_count += 1
            # skip some frames according to the frame_rate
            if f_count % frame_rate == 0:
                # compute the color histogram
                h1 = cv2.calcHist([cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)], [0], None, [256], [0,256])
                # use 1 - correlation as the metrics of distance
                tmp = 1-cv2.compareHist(h0, h1, cv2.HISTCMP_CORREL)
                if in_shot:
                    # if in shot and distance > threshold
                    # i.e. the current shot ends
                    if tmp > threshold:
                        in_shot = False
                        shot_e = f_count
                        # remove shots that are too short
                        if shot_e - shot_s > min_inter:
                            shot_list.append((shot_s, shot_e))
                            shot_count += 1
                            print("\nShot#{:<4}\t{} -- {}".format(shot_count, shot_s, shot_e))
                            # select frames with top-k keywords num
                            sorted_buf = sorted(frame_buf, key=lambda x:x[1], reverse=True)
                            if len(sorted_buf) > K:
                                sorted_buf = sorted_buf[:K]
                            # re-sort frames by frame#
                            tmp_keyframes = sorted(sorted_buf, key=lambda x:x[0])
                            for (fnum, kwnum, img, kw_list) in tmp_keyframes:
                                img_path = os.path.join(img_output_dir, "{}.jpg".format(fnum))
                                cv2.imwrite(img_path, img)
                                fnum2kwlist[fnum] = kw_list
                                print("{}:{}".format(fnum, kw_list))
                            res_key_frames += tmp_keyframes
                    # if in shot and distance < threshold
                    # i.e. the current shot continues
                    else:
                        if f_count < cur_sub[0]:
                            if not outsub_skip:
                                frame_buf.append((f_count, 0, f1, []))
                                outsub_skip = True
                        elif f_count < cur_sub[1]:
                            if not insub_skip:
                                tmp_keywords = self.sub_keywords[cur_sub_idx]
                                frame_buf.append((f_count, len(tmp_keywords), f1, tmp_keywords))
                                insub_skip = True
                        else:
                            if cur_sub_idx < len(self.sub_range) - 1:
                                cur_sub_idx += 1
                                cur_sub = self.sub_range[cur_sub_idx]
                                outsub_skip = False
                                insub_skip = False
                # if not in shot and distance < threshold
                # i.e. a new shot starts
                elif not in_shot and tmp < threshold:
                    in_shot = True
                    outsub_skip = False
                    insub_skip = False
                    shot_s = f_count
                    # flush frame buffer
                    frame_buf = []
                h0 = h1
        # deal with the last possible shot
        if in_shot:
            shot_e = f_count
            if shot_e - shot_s > min_inter:
                shot_list.append((shot_s, shot_e))
                shot_count += 1
                print("\nShot#{:<4}\t{} -- {}".format(shot_count, shot_s, shot_e))
                # select frames with top-k keywords num
                sorted_buf = sorted(frame_buf, key=lambda x:x[1], reverse=True)
                if len(sorted_buf) > K:
                    sorted_buf = sorted_buf[:K]
                # re-sort frames by frame#
                tmp_keyframes = sorted(sorted_buf, key=lambda x:x[0])
                for (fnum, kwnum, img, kw_list) in tmp_keyframes:
                    img_path = os.path.join(img_output_dir, "{}.jpg".format(fnum))
                    cv2.imwrite(img_path, img)
                    fnum2kwlist[fnum] = kw_list
                    print("{}:{}".format(fnum, kw_list))
                res_key_frames += tmp_keyframes

        fps = f_count/(time.time()-start)
        print("Avg FPS={}".format(fps))
        self.key_frames = res_key_frames
        pickle.dump(fnum2kwlist, open(os.path.join(output_dir, "fnum2kwlist.pkl"), "wb"))
        print("fnum2kwlist.pkl saved.")
