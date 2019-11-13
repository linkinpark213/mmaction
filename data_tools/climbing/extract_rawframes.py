import os
import cv2
import tqdm
import logging
import argparse


def extract_rawframes(video_path):
    extention = video_path.split('.')[-1]
    rawframes_dir = video_path[:-(len(extention) + 1)]
    os.mkdir(rawframes_dir)
    capture = cv2.VideoCapture(video_path)
    progress = tqdm.tqdm(total=cv2.CAP_PROP_FRAME_COUNT)
    n = 0
    while True:
        n += 1
        progress.update(1)
        ret, frame = capture.read()
        if not ret:
            break
        cv2.imwrite('{}/{}.png'.format(rawframes_dir, n), frame)
    logging.info('Extracted {} raw frames from video file {}'.format(n, video_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('videos_dir', type=str, help='Path to the video files')
    args = parser.parse_args()

    for filename in os.listdir(args.videos_dir):
        if filename.split('.')[-1] == 'mp4':
            extract_rawframes(os.path.join(args.videos_dir, filename))
