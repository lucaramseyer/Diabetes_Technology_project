# Diabetes_Technology_project

Hello

General information:
1. The jupyter scripts CustomNet and ResNet50 contain the code to train and evaluate models on a given dataset. 
2. The cnn python file contains all the functions and classes to run the code above.


These steps must be followed to run our code:
1. The data: The notebook files need to be provided with the dataset in the following form: A folder with all the training images named with number of image in the respective class and the name of the class. e.g. 0_bread or 0_noodles-pasta. Same for the evaluation and validation images.
2. The CustomImageDataset class in the cnn python file needs a csv file for each of the three data folders (training, evaluation, validation) with the names of the images in the first row and the class encoded as number (0=bread, 1=daryproducts, 2=dessert, ...) in the second row.
3. The jupyter scipts (CustomNet & ResNet50) can now be run to train a model
4. You can also load the weights of a model trained before (skip the train cell in the scripts). However, the models that we used to produce our results are to big to upload them to GitHub. If you need them, please approach us and we will find a way to give them to you.

If you have questions don't hesitate to ask us!
