# nuclei
nuclei kaggle challenge

# TODOs:

- k-fold CV
- weighting of labels
- accuracy calculation
- data augmentation (flip, distortion, also darken? some images are very dark where you can barely see the cells)
- cleaning up mask data (holes for ex)
- **DONE** more work on contour labels (dcan paper talks about running a disk filter)
- Tensorboard metrics
- Running in CloudML
- Evaluation
- Overlap-tile stitching or some other strategy to deal with image size differences
- Validation run does random cropping so it's not the same each time, should make more consistent
- Freeze weights in resnet - see slim nets trainer for reference
- Use custom downsample since resnet appears to converge quickly, it might be overkill
- Better way to decay learning rate (see slim net code)
- Validate training data based on thread at competition site
- Do something with focus? dcan project has a preprocessing step that measure the focus of each image.
- See file on different class modalities. Should group batches in these modes to see how mode does  in each one.
- "Also when I'm selecting crops to feed to the net at training time, I arrange to give it crops that contain an 
edge at least half the time." https://www.kaggle.com/c/data-science-bowl-2018/discussion/47690
- http://scikit-image.org/docs/dev/auto_examples/segmentation/plot_watershed.html
- See here https://github.com/tensorlayer/tensorlayer/blob/master/tensorlayer/layers.py. An upsampling layer which 
just uses tensorflow resize. Back prop flows through but are there learnable weights involved?
- Pyramid the larger images? See vooban link for another link on doing that
