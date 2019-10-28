import argparse
import cv2
import numpy as np
import torch
import mmcv
from mmcv.runner import load_checkpoint, parallel_test, obj_from_dict
from mmcv.parallel import scatter, collate, MMDataParallel

from mmaction import datasets
from mmaction.datasets import build_dataloader
from mmaction.models import build_recognizer, recognizers
from mmaction.core.evaluation.accuracy import (softmax, top_k_accuracy,
                                               mean_class_accuracy)


def inference(model, images):
    result = model([1], None, return_loss=False, img_group_0=torch.Tensor(images))
    return result


def single_test(model, data_loader):
    model.eval()
    results = []
    dataset = data_loader.dataset
    # prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            # result = model(return_loss=False, **data)
            result = inference(model, data['img_group_0'])
        results.append(result)

        # batch_size = data['img_group_0'].data[0].size(0)
        # for _ in range(batch_size):
        #     prog_bar.update()
    return results


def _data_func(data, device_id):
    data = scatter(collate([data], samples_per_gpu=1), [device_id])[0]
    return dict(return_loss=False, rescale=True, **data)


def parse_args():
    parser = argparse.ArgumentParser(description='Test an action recognizer')
    parser.add_argument('video_path', help='path to test video')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--gpus', default=1, type=int, help='GPU number used for testing')
    parser.add_argument(
        '--proc_per_gpu',
        default=1,
        type=int,
        help='Number of processes per GPU')
    parser.add_argument('--use_softmax', action='store_true',
                        help='whether to use softmax score')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    print(args.video_path)
    cap = cv2.VideoCapture(args.video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print('Video length: ', frame_count)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    buf = np.empty((frame_count - 1, 224, 224, 3), np.dtype('uint8'))

    fc = 0
    ret = True

    while (fc < frame_count - 1 and ret):
        ret, frame = cap.read()
        buf[fc] = cv2.resize(frame, (224, 224))
        fc += 1
    print('Video loaded')

    cap.release()

    buf = buf.transpose((0, 3, 1, 2))
    buf = np.expand_dims(buf, 0)
    buf = buf.astype(np.float32) - 128

    cfg = mmcv.Config.fromfile(args.config)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.data.test.test_mode = True

    if cfg.data.test.oversample == 'three_crop':
        cfg.model.spatial_temporal_module.spatial_size = 8

    model = build_recognizer(
        cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
    load_checkpoint(model, args.checkpoint, strict=True)

    outputs = inference(model, buf)

    print(outputs)
    print(np.argmax(outputs, axis=1))


if __name__ == '__main__':
    main()
