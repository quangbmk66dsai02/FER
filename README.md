
# A. Installation Guide

Our final project codes are in the form of notebooks with Python Language. There are 2 main ways to run them: Jupyter Notebook or Kaggle. We highly recommend you  run it on Kaggle to utilize Kaggle GPU for much faster training. In this guide, there are 2 sections for Kaggle version and Jupyter Notebook version respectively. Also, there is a version of code which imports already-trained model for just testing only.

Dataset:
https://drive.google.com/drive/folders/17Le8PRI5rvKkhVdwx-4Y-BToGFFjlsW-?usp=drive_link

Note that due to the large size of the dataset, we cannot upload it with our code. This dataset is needed when you run the JupyterNotebook Version (not necessary for the Kaggle version).

After downloaded and extracted, your dataset directory should look like this:

   FER\fer-2013\fer2013\fer2013.csv.

If you have any trouble during the installation process, please contact: quangbm1812@gmail.com
# 1. Kaggle version

    1. Go to https://www.kaggle.com/ and create your account.
    2. Create a new Notebook.
    3. Import "final-notebook-kaggle-version.ipynb" to your newly created notebook.
    4. Download Fer-2013 dataset from https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data.
    5. Create a new dataset with name "fer-2013" by uploading the dataset you have just downloaded. 
    6. You are ready to run the code.

    Alternatively, you can use this notebook https://www.kaggle.com/code/biminhquang/final-notebook-kaggle-version.

# 2. Jupyter Notebook
    1. You need to install Jupyter Notebook first 
    2. Make sure to install all required modules: panda, cv2, scikitlearn, ...
    3. Open file "final-notebook-jupyter-notebook.ipynb"
    4. You need to make sure that the dataset is in the correct directory.
    5. You are ready to run the code

# 3. Pretrained Model for Prediction (Fast - No training required)
    1. Open notebook "test_model.ipynb" in Jupyter Notebook.
    2. Put the processed image in the "extra-sample" folder.
    3. Change the directory to your in desired image in the "Importing test image" section.
    4. Run the code and the result is at "Test Result" section.

    Note: 
    The images should only contain human faces because of low resolution of the image already.
    The image must be in the gray-scaled form.

# B. Running Guide.
This section helps users to run the code to get the desired result as in the "Experiment and Result Analysis" section in our report.
This section applies to both notebooks running on Kaggle or Jupyter Notebook.

You will always need to run all the sections in the code ("Library", "Extract data from dataset ...", "Convert Data to RBG ... ", ...) except for "Saving model" and "Result Visualization" (they are optional).

To get the result that is most identical to those in our report, we recommend running each experiment 5 times as we did in our project.

All the results except for those related to Gridsearch are from the test set (X_test and y_test), those results related to Gridsearch are from the validation set (X_valid and y_valid).

# 1. Optimal accuracy.

Keep all the codes the same.

# 2. Without Data Normalization.

In the "Convert Data to RBG (compatible with VGG 19), also normalization using Min Max Scale" section, You need to find and remove the line:

    img_features = img_features/255 

# 3. Without Data Augmentation.

In the "Compiling and Training Model" section, you need to find the code block 





    history = model.fit(train_datagen.flow(X_train, 
                                            y_train, 
                                            batch_size = batch_size),
                                            validation_data = (X_valid, y_valid),
                                            steps_per_epoch = len(X_train) / batch_size,
                                            epochs = epochs,
                                            callbacks = callbacks,
                                            use_multiprocessing = True) 
and replace it with



    history = model.fit(X_train, y_train, batch_size = batch_size,
                                        validation_data = (X_valid, y_valid),
                                        steps_per_epoch = len(X_train) / batch_size,
                                        epochs = epochs,
                                        callbacks = callbacks,
                                        use_multiprocessing = True)

# 4. Fine-tuning different layers of VGG.

In the "Download and import weight from VGG model" section, you need to find code block:


    for layer in model.layers:
        layer.trainable = True

    for layer in model.layers:
        print(layer.name, layer.trainable) 

and replace it with:

    # Freeze layers up to block1_pool
    for layer in model.layers:
        layer.trainable = True
    for layer in model.layers[:4]:
         layer.trainable = False
    for layer in model.layers:
        print(layer.name, layer.trainable)

or:

    # Freeze layers up to block3_pool
    for layer in model.layers:
        layer.trainable = True
     for layer in model.layers[:12]:
         layer.trainable = False
    for layer in model.layers:
        print(layer.name, layer.trainable)

# 5. Focal Loss.

The code for focal loss function is in the notebook already, you only need to make a small modification.

In the "Compiling and Training Model" section, find the line:

    model.compile(loss = 'categorical_crossentropy',

and replace it with:

    model.compile(loss= focal_loss(gamma=gamma),

Note: you can change the value of gamma in the "Focal Loss" section in the code.

# 6. Weigh Initialization.

In the final code, the weights of CNN blocks are copied from the VGG19 model. If you want to perform with random weight Initialization, in the section "Download and import weight from VGG model", you need to find the line:

    vgg = tf.keras.applications.VGG19(weights = 'imagenet',

and replace it with:

    vgg = tf.keras.applications.VGG19(weights = None,


# 7. Grid search 

Learning rate and batch size grid search share a similar code structure which is also very similar to the original code. 

Notebook for learning rate:
https://www.kaggle.com/code/biminhquang/notebook-gridsearch-lr/notebook

Notebook for batch size:
https://www.kaggle.com/code/biminhquang/notebook-gridsearch-batch-size

If you want to change the Learning rate or batch size, you will need to modify code in the "Building 3 separate model for 3 batch_size setting" section. 

    learning_rates = [0.0003, 0.00005, 0.0001]

or 

    batch_size = [2,32, 2048]






