File containing all the references and progress on the creation of the backend portion of the project.

For the main backend code and architecture im using [Python](https://www.python.org/)

---- audio-processing file references ----
A lot of the initial audio processing code is pretty similar (if not identical) to the previous version of the project, so that eliviates quite a bit of work at the starting stages atleast, most of the tryharding will likely come with the model training since itll be with more species this time

The original 2 datasets used fior the initial 260 classes when creating the project are:
[Xeno-Canto Bird Recordings Extended (A-M)](https://www.kaggle.com/datasets/rohanrao/xeno-canto-bird-recordings-extended-a-m)
[Xeno-Canto Bird Recordings Extended (N-Z)](https://www.kaggle.com/datasets/rohanrao/xeno-canto-bird-recordings-extended-n-z)
Later on the projects lifecycle, the dataset will be expanded with more species

Considering the N-Z collection's CSV file is wrong from the Kaggle download, ill be using the eBird taxonomy CSV file for the corresponding bird class to the code from the xeno canto  [eBird Taxonomy v2025](https://www.birds.cornell.edu/clementschecklist/introduction/updateindex/october-2025/2025-citation-checklist-downloads/)

For the corresponding bird classes to be able to add photos/location ill be using the iNaturalist endpoint so i can access the photos, but that will come later on.

Deciding on the amount of used cores for preprocessing - [How the "Number of Workers" Parameter in PyTorch DataLoader Actually Works](https://www.geeksforgeeks.org/deep-learning/how-the-number-of-workers-parameter-in-pytorch-dataloader-actually-works/)

res_type(soxr_hq is chosen) used alongside with the librosa library used for the resampling: [librosa.resample](https://librosa.org/doc/main/generated/librosa.resample.html). Its high quality (second highest behind VHQ, and is also much faster for the resampling as we are handling 500+ classes).

For the console progress when training, processing audios etc [Tqdm](https://www.datacamp.com/tutorial/tqdm-python) will be used for the progress bars, as its accurate and lightweight:
To be able to match the .wav and .mp3 types of files, [Globs](https://gulpjs.com/docs/en/getting-started/explaining-globs/) will be used in the audio process classes function


---- data set splitting file ----
[The Importance of Splitting Datasets into Training, Validation, and Test Sets](https://ruveydakardelcetin.medium.com/the-importance-of-splitting-datasets-into-training-validation-and-test-sets-417caaeae91d)

Creating a seperate data split file. The old project split the data in the exact same file as it was training (the splitting itself took place right before the training) and that took a lot of processing time even tho splitting the data into seperate sets is a one time thing when training a model. Having it happen before the model training allows the training to go faster in terms of speed and also use much less memory.
The data splitting function now also shows the files that each split has, aswell as which classes have too little samples, that way while testing it i can know going forward which classes will need expanding.


---- data resizing file ----


---- data parser file ----
The data parser idea is to pass only one class at a time while training the model. That way its lighter and easier to keep track of the seperate steps of the model creating process


---- model training file references ----
[choosing between resnet vs efficientnet](https://medium.com/@enrico.randellini/image-classification-resnet-vs-efficientnet-vs-efficientnet-v2-vs-compact-convolutional-c205838bbf49)
Ultimately going for EfficientNet as its a bit better and ultimately much lighter to use

[How to Convert a TensorFlow Model to PyTorch?](https://www.geeksforgeeks.org/deep-learning/how-to-convert-a-tensorflow-model-to-pytorch/)

[How to use GPU acceleration in PyTorch?](https://www.geeksforgeeks.org/deep-learning/how-to-use-gpu-acceleration-in-pytorch/)

[How I changed CPU to GPU support for my ML Model (Easy Guide)](https://medium.com/@vsquarevaibhavverma/how-i-enabled-gpu-support-for-my-ml-model-easy-guide-ready-f5f455358d6d)


---- add samples file ----
A dedicated file who's purpose is to extract more classes/more samples for a chosen class. It uses the Xeno Canto v3 API and a dedicated custom API key.
It has a limit of downloads per request, max of 100 recordings and it only downloads recordings with a A or B rating.

---- backend file references ----