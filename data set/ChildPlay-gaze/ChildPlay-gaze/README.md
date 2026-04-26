
# ChildPlay Dataset


**Paper**

ChildPlay: A New Benchmark for Understanding Children’s Gaze Behaviour (Tafasca et al. ICCV 2023)

**Abstract**

Gaze behaviors such as eye-contact or shared attention are important markers for diagnosing developmental disorders in children. While previous studies have looked at some of these elements, the analysis is usually performed on private datasets and is restricted to lab settings. Furthermore, all publicly available gaze target prediction benchmarks mostly contain instances of adults, which makes models trained on them less applicable to scenarios with young children. In this paper, we propose the first study for predicting the gaze target of children and interacting adults. To this end, we introduce the ChildPlay Gaze dataset: a curated collection of short video clips featuring children playing and interacting with adults in uncontrolled environments (e.g. kindergarten, therapy centers, preschools etc.), which we annotate with rich gaze information. Our results show that looking at faces prediction performance on children is much worse than on adults, and can be significantly improved by fine-tuning models using child gaze annotations.

**Dataset Description**

The ChildPlay Gaze dataset is composed of 401 clips extracted from 95 longer YouTube videos, totaling 120549 frames. For each clip, we select up to 3 people, and annotate all of them in each frame (when they are visible) with gaze information.

The `annotations` folder contains 3 subfolders: `train`, `val` and `test`. Each subfolder contains csv annotation files in the format `videoid_startframe_endframe.csv`. In this naming convention, `videoid` refers to the original video from which the clip was extracted, while the `startframe` and `endframe` refer to the starting and ending frames of the clip in the original YouTube video `videoid`. 

For example, one of the original videos downloaded from YouTube is  `1Ab4vLMMAbY.mp4` where `1Ab4vLMMAbY` is the YouTube video ID, which can be used directly to build the URL (i.e. https://www.youtube.com/watch?v=1Ab4vLMMAbY). The annotation file `1Ab4vLMMAbY_2354-2439.csv` found under `ChildPlay/annotations/test` refers to the annotation of the clip `1Ab4vLMMAbY_2354-2439.mp4` extracted from `1Ab4vLMMAbY.mp4` which starts at frame `2354` and ends at frame `2439` (included). The numbering starts from 1.

Please note that some videos were recorded at 60 FPS, whereas most are at 24-30. When we extracted clips from these, we also downsampled to 30 FPS by skipping every other frame. The starting and ending frames in their names correspond to their numbers in the original video, but we also include the mention `downsampled` so they are recognizable. For example, `smwfiZd8HLc_7508-8408-downsampled.mp4` is a clip extracted between frames 7508 and 8408 from the video `smwfiZd8HLc.mp4`. However, it only contains 451 frames as opposed to the expected 901 = 8408 - 7508 + 1 since it was downsampled.

Each annotation csv file has one row per annotated person per frame, and includes the following columns:
- **clip:** the name of the clip (without extension). This value is duplicated across the entire dataframe.
- **frame:** the relative frame number in the clip. For example, frame `n` refers to the nth frame of the clip. If the clip is named `videoid_start-end`, then frame `n` in the clip corresponds to frame `start + n - 1` in the original video (unless the clip was downsampled).
- **person_id:** an id used to separate and track annotated people in the clip.
- **bbox_x:** the x-value of the upper left corner of the head bounding box of the person (in the image frame).
- **bbox_y:** the y-value of the upper left corner of the head bounding box of the person (in the image frame).
- **bbox_width:** the width of the head bounding box of the person (in the image frame).
- **bbox_height:** the height of the head bounding box of the person (in the image frame).
- **gaze_class:** a gaze label to determine the type of gaze behavior. One of `[inside_visible, outside_frame, gaze_shift, inside_occluded, inside_uncertain, eyes_closed]`. Refer to the paper for the definitions of these flags. The labels `inside_visible` and `outside_frame` in particular, correspond to the standard inside vs outside label found in other gaze following datasets (e.g. GazeFollow and VideoAttentionTarget)
- **gaze_x:** the x-value of the target gaze point (in the image frame). This value is set to -1 when `gaze_class != inside_visible`. 
- **gaze_y:** the y-value of the target gaze point (in the image frame). This value is set to -1 when `gaze_class != inside_visible`. 
- **is_child:** a binary flag denoting whether the person is a child or an adult.

You will also find a `videos.csv` file containing a list of videos to download, from which the ChildPlay clips were extracted, along with other metadata (e.g. channel ID, fps, resolution, etc.). There is also a `clips.csv` file containing similar information but for each clip, and a `splits.csv` detailing the train/val/test split of each clip.

Furthermore, we provide utility scripts to extract the necessary clips and images from the videos (assuming you have already downloaded them).

**Dataset Acquisition**

Please follow the steps below to setup the dataset:

1. Download the 95 original videos listed in `videos.csv` from YouTube. You can use the python package `pytube` or some other tool.
2. Use the `extract-clips-from-videos.py` script to extract both the clips and corresponding frames from the videos. In order to use the script, you have to supply the following flags `--clip_csv_path` (path to the `clips.csv` file), `--video_folder` (path to the folder of the downloaded videos), `--clip_folder` (path where to save the clips, it will be created if it doesn't exist), `--image_folder` (path where to save the images, it will be created if it doesn't exist). Please note that you need to have the packages `pandas`, `tqdm` and `opencv` installed. The script also requires `ffmpeg` for the extraction.

The final dataset folder structure should look like

```
.
├── annotations
│   ├── test
│   │   ├── 1Ab4vLMMAbY_2354-2439.csv
│   │   ├── ...
│   ├── train
│   │   ├── 1Aea8BH-PCs_1256-1506.csv
│   │   ├── ...
│   ├── val
│   │   ├── bI1GohGXSt0_2073-2675.csv
│   │   ├── ...
├── clips
│   ├── 1Ab4vLMMAbY_2354-2439.mp4
├── images
│   ├── 1Ab4vLMMAbY_2354-2439
│   │   ├── 1Ab4vLMMAbY_2354.jpg
│   │   ├── ...
│   ├── ...
├── videos
│   ├── 1Ab4vLMMAbY.mp4
│   ├── ...
├── clips.csv
├── README.md
├── extract-clips-from-videos.py
├── splits.csv
└── videos.csv
```

**Contact**

Please reach out to Samy Tafasca (samy.tafasca@idiap.ch) or Jean-Marc Odobez (odobez@idiap.ch) if you have any questions, or if some videos are no longer available on YouTube.