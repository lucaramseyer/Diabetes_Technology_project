# Diabetes_Technology_project

Open Questions:
  Can we simply calculate the memory reqirement of our model based on the layers?
  Sould we use different pretrained models? (we used RESnet50 so far)
  The pretrained model doesn't seem to have huge memory requirements whereas our model quickly needs lots of memory. Should we try to minimize the memory requiremets of our cnn?
  HPO?
  Does a bigger batch size always mean better performance?
  Is the train loss the mean loss over all batches in an epoch?
  Why exactly do we need a validation set? can't we just use the training set?
  


ToDo:
  Add classifier layer to pretrained model
  Evaluate performance of pretrained model
  Add learining rate plot to our code
  Determine memory requirements of pretrained model
  Activation maps!!
