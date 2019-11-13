import os
import cv2
import tqdm
import logging
import argparse


def range_overlap_adjust(list_ranges):
    overlap_corrected = []
    for start, stop in sorted(list_ranges):
        if overlap_corrected and start - 1 <= overlap_corrected[-1][1] and stop >= overlap_corrected[-1][1]:
            overlap_corrected[-1] = min(overlap_corrected[-1][0], start), stop
        elif overlap_corrected and start <= overlap_corrected[-1][1] and stop <= overlap_corrected[-1][1]:
            break
        else:
            overlap_corrected.append((start, stop))
    return overlap_corrected


def extract_rawframes(video_path, frame_ranges):
    logging.info('')
    logging.info('Extracting frames from {}'.format(video_path))
    extention = video_path.split('.')[-1]
    rawframes_dir = video_path[:-(len(extention) + 1)]
    if not os.path.isdir(rawframes_dir):
        os.mkdir(rawframes_dir)
    capture = cv2.VideoCapture(video_path)
    n = 0
    range_pointer = 0
    start_milestones = [frame_range[0] for frame_range in frame_ranges]
    end_milestones = [frame_range[1] for frame_range in frame_ranges]
    write = False
    with tqdm.tqdm(total=cv2.CAP_PROP_FRAME_COUNT) as pbar:
        while True:
            n += 1
            if range_pointer == len(start_milestones):
                break
            if n == start_milestones[range_pointer]:
                logging.info('Starting writing at frame #{}'.format(start_milestones[range_pointer]))
                write = True
            if n == end_milestones[range_pointer] + 1:
                logging.info('Stopping writing at frame #{}'.format(end_milestones[range_pointer]))
                write = False
                range_pointer += 1
            pbar.update(1)
            ret, frame = capture.read()
            if not ret:
                break
            if write:
                cv2.imwrite('{}/{}.png'.format(rawframes_dir, n), frame)
    logging.info('Extracted {} raw frames from video file {}'.format(n, video_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Raw frame extractor')
    parser.add_argument('annotation_file', type=str, help='Path to the annotation file')
    parser.add_argument('videos_dir', type=str, help='Path to the video files')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    annotation_file = open(args.annotation_file, 'r')

    current_video_name = None
    frame_ranges = []
    for index, line in enumerate(annotation_file):
        video_name, target_id, start_frame, end_frame = line.strip().split()
        target_id, start_frame, end_frame = int(target_id), int(start_frame), int(end_frame)
        if video_name != current_video_name:
            # Extract frames
            if current_video_name is not None:
                frame_ranges = range_overlap_adjust(frame_ranges)
                extract_rawframes(os.path.join(args.videos_dir, current_video_name + '.mp4'), frame_ranges)

            current_video_name = video_name
            frame_ranges = [(start_frame, end_frame)]
        else:
            # Wait for all ranges are collected
            frame_ranges.append((start_frame, end_frame))

    # Extract last video
    if current_video_name is not None:
        frame_ranges = range_overlap_adjust(frame_ranges)
        extract_rawframes(os.path.join(args.videos_dir, current_video_name + '.mp4'), frame_ranges)
