# nuclei
nuclei kaggle challenge

# TODOs:

- k-fold CV
- weighting of labels
- accuracy calculation
- data augmentation (flip, distortion, also darken? some images are very dark where you can barely see the cells)
- cleaning up mask data (holes for ex)
- more work on contour labels (dcan paper talks about running a disk filter)
- Tensorboard metrics
- Running in CloudML
- Evaluation
- Overlap-tile stitching or some other strategy to deal with image size differences
- Validation run does random cropping so it's not the same each time
- Freeze weights in resnet - see slim nets trainer for reference
- Use custom downsample since resnet appears to converge quickly, it might be overkill
- Better way to decay learning rate (see slim net code)
- Validate training data based on thread at competition site
- Do something with focus? dcan project has a preprocessing step that measure the focus of each image.
- Pyramid the larger images? See vooban link for another link on doing that
