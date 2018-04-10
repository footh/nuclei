# nuclei
nuclei kaggle challenge

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