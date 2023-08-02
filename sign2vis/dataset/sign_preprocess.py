# preprocess sign videos
import os
import sys
sys.path.append('../')
sys.path.append('../../')
import numpy as np
import cv2
import json

# WIKISQL_VIDEO_PATH = '/home/huangwencan/data/sign2sql/'
# VIDEO_ROOT_LIST = [os.path.join(WIKISQL_VIDEO_PATH, 'length%d'%i)  for i in range(3, 6+1)]

VIDEO_PATH = '/mnt/silver/guest/zgb/Sign2Vis/new_pose_format'
TARGET_DIR = "/mnt/silver/guest/zgb/Sign2Vis/new_npy_data"


def handle_frame(frame):
    x = 255 - frame  # reverse color
    kernel = np.ones((3, 3), dtype=np.uint8)
    dilate = cv2.dilate(x, kernel, 10)  # dilation, #iteration=10
    x2 = cv2.resize(dilate, (144, 144))  # resize to 144, can be 108
    x2 = np.mean(x2/255, axis=-1)  # to gray
    return x2  # (144, 144)


def handle_video(video_file_path):
    cap = cv2.VideoCapture(video_file_path)
    vid_array = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        x = handle_frame(frame)
        vid_array.append(x)
    vid_array = np.asarray(vid_array)
    return vid_array  # (N, 144, 144)


# def check(video_root):
#     save_root = video_root+'_preprocessed'
#     cnt = 0
#     for vid_file in os.listdir(video_root):
#         save_path = os.path.join(save_root, vid_file[:-4])
#         if not os.path.exists(save_path+'.npy'):
#             print('Not Exist: '+save_path+'.npy')
#             cnt += 1
#             # handle
#             vid_array = handle_video(os.path.join(video_root, vid_file))
#             np.save(save_path, vid_array)
#     print('NEW', cnt)


if __name__ == '__main__':
    # for video_root_name in VIDEO_ROOT_LIST:
    #     video_root = os.path.join(WIKISQL_VIDEO_PATH, video_root_name)
    #     check(video_root)
    video_lengths = []
    cnt = 0
    video_root = VIDEO_PATH
    print('processing start')
    pre_data = ['888@x_name@ASC', '888@x_name@DESC', '2417@x_name@ASC', '2417@x_name@DESC', '2417@y_name@DESC', '895@x_name@DESC', '888@y_name@ASC', '888@y_name@DESC', '2417@y_name@ASC']
    # with open('/mnt/silver/guest/zgb/Sign2Vis/Text2Pose/last_trans.jsonl', 'r', encoding='utf-8') as f:
    #     for line in f.readlines():
    #         pre_data.append(json.loads(line.strip())['id'])
    for id in pre_data:
        # print(id + '.mp4')
    # for vid_file in os.listdir(video_root):
        # print(vid_file)
        vid_file = id + '.mp4'
        vid_array = handle_video(os.path.join(video_root, vid_file))
        video_lengths.append(vid_array.shape[0])
        # save_root = video_root+'_preprocessed'
        save_root = TARGET_DIR
        if not os.path.exists(save_root):
            os.makedirs(save_root)
        save_path = os.path.join(save_root, vid_file[:-4])
        print(vid_array.shape)
        np.save(save_path, vid_array)
        cnt += 1
        print('processed %d videos' % cnt)
        if cnt % 100 == 0:
            print('processed %d videos' % cnt)
    print('video length statistics:')
    print(np.min(video_lengths), np.mean(video_lengths), np.max(video_lengths)) #

