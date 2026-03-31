#!/usr/bin/env bash
set -eo pipefail

for backbone in "resnet18" "resnet34" "mobilenetv2"; do
  for cls in "all" "bird" "cat" "cat_like" "dog" "dog_like" "horse_like" "small_animals"; do
    PYTHONPATH='/mnt/data/afarec/code/face_detection/RetinaFace/':$PYTHONPATH \
    python /mnt/data/afarec/code/face_detection/RetinaFace/train.py \
      --train-data-labels "/mnt/data/afarec/data/OAFI_full/labels_yunet/labels_${cls}_train.txt" \
      --train-data-images "/mnt/data/afarec/data/OAFI_full/images/train" \
      --weights "$(dirname "$0")/weights/retinaface_${backbone}.pth" \
      --network $backbone \
      --batch-size 8 \
      --learning-rate 1e-4 \
      --print-freq 40 \
      --save-dir "./work_dir/retinaface_${backbone}_${cls}/"

    echo "Start prediction for RetinaFace ${backbone} for class ${cls}"
    PYTHONPATH='/mnt/data/afarec/code/face_detection/RetinaFace/':$PYTHONPATH \
    python /mnt/data/afarec/code/face_detection/RetinaFace/evaluate_widerface.py \
      -w "$(dirname "$0")/work_dir/retinaface_${backbone}_${cls}/${backbone}_final.pth" \
      --network $backbone \
      --origin-size \
      --save-folder "$(dirname "$0")/work_dir/retinaface_${backbone}_${cls}/results/" \
      --dataset-folder "/mnt/data/afarec/data/OAFI_full/images/test/" \
      --dataset-labels "/mnt/data/afarec/data/OAFI_full/labels_yunet/labels_${cls}_test.txt"

    echo "Start latency test for RetinaFace ${backbone} for class ${cls}"
    PYTHONPATH='/mnt/data/afarec/code/face_detection/RetinaFace/':$PYTHONPATH \
    python /mnt/data/afarec/code/face_detection/RetinaFace/evaluate_widerface.py \
      -w "$(dirname "$0")/work_dir/retinaface_${backbone}_${cls}/${backbone}_final.pth" \
      --network $backbone \
      --origin-size \
      --save-folder "$(dirname "$0")/work_dir/retinaface_${backbone}_${cls}/results/" \
      --dataset-folder "/mnt/data/afarec/data/OAFI_full/images/test/" \
      --dataset-labels "/mnt/data/afarec/data/OAFI_full/labels_yunet/labels_${cls}_test.txt" \
      --latency_test 1000
  done

  echo "Start evaluation for RetinaFace ${backbone} for baseline"
  PYTHONPATH='/mnt/data/afarec/code/face_detection/RetinaFace/':$PYTHONPATH \
  python /mnt/data/afarec/code/face_detection/RetinaFace/evaluate_widerface.py \
    -w "$(dirname "$0")/weights/retinaface_${backbone}.pth" \
    --network $backbone \
    --origin-size \
    --save-folder "$(dirname "$0")/work_dir/retinaface_${backbone}_pretrained/results/" \
    --dataset-folder "/mnt/data/afarec/data/OAFI_full/images/test/" \
    --dataset-labels "/mnt/data/afarec/data/OAFI_full/labels_yunet/labels_all_test.txt"

  echo "Start latency test for RetinaFace ${backbone} for class pretrained"
  PYTHONPATH='/mnt/data/afarec/code/face_detection/RetinaFace/':$PYTHONPATH \
  python /mnt/data/afarec/code/face_detection/RetinaFace/evaluate_widerface.py \
    -w "$(dirname "$0")/weights/retinaface_${backbone}.pth" \
    --network $backbone \
    --origin-size \
    --save-folder "$(dirname "$0")/work_dir/retinaface_${backbone}_pretrained/results/" \
    --dataset-folder "/mnt/data/afarec/data/OAFI_full/images/test/" \
    --dataset-labels "/mnt/data/afarec/data/OAFI_full/labels_yunet/labels_all_test.txt" \
    --latency_test 1000
done
