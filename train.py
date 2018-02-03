def loss(c_logits, s_logits, c_labels, s_labels):
    """
        Derives a loss function given the contour and segment results and labels
    """
    # TODO: weighting of loss terms. DCAN project uses proportion of values that equal 1 for contours and segments
    # The DCAN paper talks about weights for the 'auxiliary classifiers'. In the FCN which the paper refers, the 
    # auxiliary classifiers are the pre-fused results from the different levels of the convnet. Should that be the same here?
    # Means there will be 6 of them - 3 for each label type.
    