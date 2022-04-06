# ActVA
Repository for IWQoS'20 paper "High-quality Activity-Level Video Advertising".

While doing this work, existing video advertising and image retrieval datasets were unable to meet our evaluation needs, since the proposed activity graph integrates both visual and textual features.
Advertisers are interested in the keywords that characters talk about in the videos, which can be extracted from the subtitle data instead of image description words. 
So we collected a dataset of videos and the matched subtitle files, which can be downloaded from [google drive](https://drive.google.com/file/d/1-3Rq1cFISSFWBg4vj-G66zaBVUuvS_L6/view?usp=sharing).

`lib/key_frame_detector.py`: implement the proposed detection method of key frames, based on both visual and textual information