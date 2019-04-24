# sketch-photo2seq

This is the reimplementation code of CVPR'2018 paper [Learning to Sketch with Shortcut Cycle Consistency](http://openaccess.thecvf.com/content_cvpr_2018/papers/Song_Learning_to_Sketch_CVPR_2018_paper.pdf).

| Photo | Generated examples |
| --- | --- |
| ![example1](https://github.com/MarkMoHR/sketch-photo2seq/blob/master/assets/QMUL/2482805404/photo_gt.png) | ![example1-sketch](https://github.com/MarkMoHR/sketch-photo2seq/blob/master/assets/QMUL/2482805404/sketch_pred_multi.svg) |
| ![example2](https://github.com/MarkMoHR/sketch-photo2seq/blob/master/assets/QMUL/2509386515/photo_gt.png) | ![example2-sketch](https://github.com/MarkMoHR/sketch-photo2seq/blob/master/assets/QMUL/2509386515/sketch_pred_multi.svg) |


## Requirements

- Python 3
- Tensorflow (>= 1.4.0)
- [InkScape](https://inkscape.org/) or [CairoSVG](https://cairosvg.org/) (For vector sketch rendering. Choose one of them is ok.)

  ```
  sudo apt-get install inkscape
  # or
  pip3 install cairosvg
  ```


## Data Preparations
From the paper, we need to pre-train the model on the [*QuickDraw*](https://github.com/googlecreativelab/quickdraw-dataset#sketch-rnn-quickdraw-dataset) dataset. So we need to preprocess both the *QuickDraw-shoes* and *QMUL-shoes* data following these steps:

1. QuickDraw-shoes

    - Download the `sketchrnn_shoes.npz` data from [*QuickDraw*](https://github.com/googlecreativelab/quickdraw-dataset#sketch-rnn-quickdraw-dataset)
    - Place the package under `datasets/QuickDraw/shoes/npz/` directory
    - run the command:
      ```
      python quickdraw_data_processing.py
      ```

1. QMUL-shoes

    - Download the photo data from [QMUL-Shoe-Chair-V2](http://www.eecs.qmul.ac.uk/~qian/Project_cvpr16.html)
    - Unzip the *ShoeV2_photo* package and place all `.png` under `datasets/QMUL/shoes/photos/` directory
    - Download the preprocessed sketch data from [here](https://drive.google.com/drive/folders/15-1NQXGFUEaQkM0EvefdTWP9tBHRZ2jp) and place the two `.h5` packages under `datasets/QMUL/shoes/` directory


## Training

1. QuickDraw-shoes pre-training

    - Change the value to `QuickDraw` in [`model.py`](https://github.com/MarkMoHR/sketch-photo2seq/blob/master/model.py)-`get_default_hparams`-`data_type`
    - run the command:
      ```
      python sketch_p2s_train.py
      ```

1. QMUL-shoes training

    - Change the value to `QMUL` in [`model.py`](https://github.com/MarkMoHR/sketch-photo2seq/blob/master/model.py)-`get_default_hparams`-`data_type`
    - Make sure the QuickDraw-shoes pre-training models/checkpoint are placed under `outputs/snapshot/` directory
    - Change the value to `True` in [`sketch_p2s_train.py`](https://github.com/MarkMoHR/sketch-photo2seq/blob/master/sketch_p2s_train.py)-`resume_training`
    - run the command:
      ```
      python sketch_p2s_train.py
      ```
      
#### Training loss

The following figure shows the total *loss*, *KL loss* and *reconstruction loss* during training with QuickDraw-shoes pre-trained within 30k iterations and the following QMUL-shoes trained within 40k iterations.

![loss](https://github.com/MarkMoHR/sketch-photo2seq/blob/master/assets/loss.png)


## Sampling

1. QuickDraw-shoes

    - Make sure the value of `data_type` to be `QuickDraw` in [`model.py`](https://github.com/MarkMoHR/sketch-photo2seq/blob/master/model.py)
    - Place models/checkpoint/config under `outputs/snapshot/QuickDraw/` directory
    - run the command:
      ```
      python sketch_p2s_sampling.py
      ```

1. QMUL-shoes

    - Make sure the value of `data_type` to be `QMUL` in [`model.py`](https://github.com/MarkMoHR/sketch-photo2seq/blob/master/model.py)
    - Place models/checkpoint/config under `outputs/snapshot/QMUL/` directory
    - run the command:
      ```
      python sketch_p2s_sampling.py
      ```

All results can be found under `outputs/sampling/` dir.



## Credits
- This code is largely borrowed from repos [Sketch-RNN](https://github.com/tensorflow/magenta/tree/master/magenta/models/sketch_rnn) and [deep_p2s](https://github.com/seindlut/deep_p2s).





