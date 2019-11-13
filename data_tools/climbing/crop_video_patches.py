import os
import cv2
import logging
import argparse
import numpy as np


def fill_gaps(track, target_id):
    logging.info('Gap-filling: Length before filling: {}'.format(len(track)))
    output_trajectory = []
    current_frame = -1
    for i in range(len(track)):
        if current_frame > -1 and track[i][0] - current_frame > 1:
            box_gap = track[i][2:6] - output_trajectory[-1][2:6]
            frame_gap = track[i][0] - current_frame
            logging.info('Gap found between frame {} and frame {}'.format(current_frame, track[i][0]))
            unit = box_gap / frame_gap
            logging.info('Gap unit: ({:.3f}, {:.3f}, {:.3f}, {:.3f})'.format(unit[0], unit[1], unit[2], unit[3]))
            for j in range(current_frame + 1, int(track[i][0])):
                to_fill = output_trajectory[-1][2:6] + unit
                output_trajectory.append(np.array([j, target_id, to_fill[0], to_fill[1], to_fill[2], to_fill[3]]))
        current_frame = int(track[i][0])
        output_trajectory.append(track[i])
    logging.info('Gap-filling: After filling: {}'.format(len(output_trajectory)))
    return np.array(output_trajectory)


def smooth(data, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    smoothed = np.array([np.convolve(data[:, i], window) for i in range(2, 6)]).transpose((1, 0))[
               int(window_size / 2): int(-window_size / 2), :]
    data = np.concatenate((data[:, 0:1], data[:, 1:2], smoothed), axis=1)
    return data


def crop(img, x_c, y_c, window_size):
    x_base = 0
    y_base = 0
    padded_img = img
    if x_c < window_size / 2:
        padded_img = cv2.copyMakeBorder(img, 0, 0, window_size / 2, 0, borderType=cv2.BORDER_REFLECT)
        x_base = window_size / 2
    if x_c > img.shape[1] - window_size / 2:
        padded_img = cv2.copyMakeBorder(img, 0, 0, 0, window_size / 2, borderType=cv2.BORDER_REFLECT)
    if y_c < window_size / 2:
        padded_img = cv2.copyMakeBorder(img, window_size / 2, 0, 0, 0, borderType=cv2.BORDER_REFLECT)
        y_base = window_size / 2
    if y_c > img.shape[0] - window_size / 2:
        padded_img = cv2.copyMakeBorder(img, 0, window_size / 2, 0, 0, borderType=cv2.BORDER_REFLECT)

    return padded_img[int(y_base + y_c - window_size / 2), int(y_base + y_c + window_size / 2),
           int(x_base + x_c - window_size / 2): int(x_base + x_c + window_size / 2), :]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Video patch cropper')
    parser.add_argument('annotation_file', type=str, help='Path to the annotation file')
    parser.add_argument('rawframes_dir', type=str, help='Path to the extracted raw frame files')
    parser.add_argument('track_dir', type=str, help='Path to the tracked trajectory files')
    parser.add_argument('--video_size', type=int, required=False, default=256)
    parser.add_argument('--output_dir', type=str, required=False, default='out')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    if not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
    annotation_file = open(args.annotation_file, 'r')

    for line in annotation_file:
        video_name, target_id, start_frame, end_frame = line.strip().split()
        target_id, start_frame, end_frame = int(target_id), int(start_frame), int(end_frame)
        video_writer = cv2.VideoWriter(
            os.path.join(args.output_dir, '{}_{}_{}_{}.mp4'.format(video_name, target_id, start_frame, end_frame)),
            cv2.VideoWriter_fourcc(*'mp4v'), 30, (256, 256))

        data = np.loadtxt(os.path.join(args.track_dir, video_name + '.txt'), delimiter=',')
        data = data[:, :6]
        data = data[data[:, 1] == target_id]
        data = data[data[:, 0] >= start_frame]
        data = data[data[:, 0] <= end_frame]

        logging.info('Filling gaps for video {} target #{}'.format(video_name, target_id))
        # np.savetxt('{}_{}_before.txt'.format(video_name, target_id), data, fmt='%.2f')
        data = fill_gaps(data, target_id)
        # np.savetxt('{}_{}_after.txt'.format(video_name, target_id), data, fmt='%.2f')

        data = smooth(data, window_size=9)

        for detection in data:
            box = detection[2:6]
            x_c, y_c = (detection[2] + detection[4]) / 2, (detection[3] + detection[5]) / 2
            w, h = detection[4] - detection[2], detection[5] - detection[3]
            img = cv2.imread(os.path.join(args.rawframes_dir, video_name, '{}.png'.format(int(detection[0]))))
            patch = crop(img, x_c, y_c, max(w, h))

            patch = cv2.resize(patch, (args.video_size, args.video_size))
            video_writer.write(patch)

        video_writer.release()
