# nuclei
Kaggle 2018 Data Science Bowl solution

Implementation based on "[DCAN: Deep Contour-Aware Networks for Accurate Gland Segmentation](https://arxiv.org/pdf/1604.02677.pdf)" paper

# Instructions

## Preprocessing
 - Data in the format provided by the competition should be in the *raw-data* directory under a given source directory (ex. *train*, *test*, etc.)
 - The H&E data with masks was used which is found on the competition site kernels. Code was run to remove overlaps of the masks (see notebooks)
 - TNBC data was used which is found at this link: https://zenodo.org/record/1175282. Code was run to get it into the format used by the competition data (see *tnbc/process.py*)
 - Mosaics need to be created by running the code in the *mosaics* directory. This code and the CSVs were provided by other competitors
 - Open up python console, import data module and run *data.setup(src='train')* (or whatever source directory is desired)
 - The above method will move the input image to the same source directory in the *model-data* directory along with a ground truth segment and contour file for every .png image found in raw-data.

## Training
 - The model uses a resnet_v1_50 downsampling path which is taken from tensorflow slim. Initial weights can be found here: https://github.com/tensorflow/models/tree/master/research/slim
 - Training can begin with a command like this:

```bash
python train.py --batch_size=20 --run_desc=bs20_resinit_01 --checkpoint_file=training-runs/init/resnet_v1_50.ckpt --checkpoint_filter=resnet_v1_50 --notes=True
```

This will initialize the run with a batch size of 20 using the pre-trained weights (note the location) with sensible hyperparameters. The *notes* argument allows for writing a quick note about the training run. Checkpoint files will be saved in the *training-runs* directory under the directory of the *run_desc* argument

 - The best model using all the data took about 6K steps before validation loss stopped increasing. The top 5 checkpoints will be saved.
 
## Evaluation
 - Evaluation and creation of the submission file is done with a command like this:

```bash
python eval.py --trained_checkpoint=training-runs/bs20_51init_03/best/dcan_vloss-0.12811-s-0.865-c-0.720.ckpt-1300*0,training-runs/bs20_resinit_10c/best/dcan_vloss-0.17724-s-0.819-c-0.702.ckpt-5600*0 --src=test --submission_file=combined_03_10c
```

The *trained_checkpoint* argument is a comma-delimited list of checkpoints (more than one will ensemble). Follow each checkpoint path with an *0. The *src* argument indicates the path of the source images (under *model-data*) and the submission file is a string describing the submission that will be added to the final name of the file

 - Submission file will be written out to the *submissions* directory

# TODOs:

- k-fold CV

- **DONE** weighting of labels
  Conclusion: per-label weighting done, still need to research weighting between segments and contours

- accuracy calculation

- **DONE** data augmentation (flip, distortion, also darken? some images are very dark where you can barely see the cells, tf.image has random brightness, darkness, hue and saturation)

- **DONE** cleaning up mask data (holes for ex)

- **DONE** more work on contour labels (dcan paper talks about running a disk filter)

- Tensorboard metrics

- Running in CloudML

- **DONE** Evaluation

- **DONE** Overlap-tile stitching or some other strategy to deal with image size differences

- **DONE** Validation run does random cropping so it's not the same each time, should make more consistent (could choose the top/left when index is built?)

- **DONE** Yielded bad score - Freeze weights in resnet - see slim nets trainer for reference

- **DONE** Didn't converge as well. Use custom downsample since resnet appears to converge quickly, it might be overkill

- **DONE** Better way to decay learning rate (see slim net code) 
  Conclusion: not sure if this should be done with ADAM optimizer. It would be easy to implement with tf.train.exponential_decay, but ADAM already does a form of exponential decay so the step version might be OK. 
  If optimizer is changed, consider using it. Maybe use exponential_decay with MomentumOptimizer (dcan project uses that)

- Validate training data based on thread at competition site

- Do something with focus? dcan project has a preprocessing step that measure the focus of each image.

- See file on different class modalities. Should group batches in these modes to see how mode does in each one.

- **DONE** "Also when I'm selecting crops to feed to the net at training time, I arrange to give it crops that contain an 
edge at least half the time." https://www.kaggle.com/c/data-science-bowl-2018/discussion/47690

- **DONE** http://scikit-image.org/docs/dev/auto_examples/segmentation/plot_watershed.html

- See here https://github.com/tensorlayer/tensorlayer/blob/master/tensorlayer/layers.py. An upsampling layer which 
just uses tensorflow resize. Back prop flows through but are there learnable weights involved?

- Pyramid the larger images? See vooban link for another link on doing that

- optimizer: https://www.quora.com/Why-do-the-state-of-the-art-deep-learning-models-like-ResNet-and-DenseNet-use-SGD-with-momentum-over-Adam-for-training 
and https://stackoverflow.com/questions/36162180/gradient-descent-vs-adagrad-vs-momentum-in-tensorflow
