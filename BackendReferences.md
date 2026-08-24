File containing all the references on the creation and of the backend portion of the project. Will contain detailed explanations on what each file does and how it does it, so i can more easily document it for the actual documentation.

For the main backend code and architecture im using [Python](https://www.python.org/)

Whoalistic structure of the backend side of the project follows this order:
Gathering all the audio files in /dataset. All the current files are taken directly from XenoCanto and are withh the highest available rating (A and B). Any more samples to each class are added with the add_samples.py file + the respective class name.
↓
Processing each of the classes in the /dataset folder and creating mel spectrograms, saved in a .npy array format (audio_processing.py).
↓
Splitting the total spectrogram set into 3 seperate sets: one for training one, for testing and one for validation (dataset_split.py).
↓
Running the training set through the augment_trainset.py file. It applies a SpecAugment filter onto the training set allowing the set to be more robust with more variation.
↓
Resizing the three splits, its done in order to make them fit the preffered EfficientNet size format (resize_splits.py).
↓
Once the training set has been created and resized, its ran through the BirdNet teacher file. The concep behind it is that the teacher will be able to learn from the dataset, backed by the already well trained and expansive BirdNet model. This way it becomes a "teacher" to the main model that gets built in the next step.
↓
Model training happens with the model_training.py and data_parser.py files respectively. The data parser is used to alleviate the memory use and GPU strain that loading the entire test/train/val sets would put on the machine. With the help of the teacher the EfficientNet model gets trained on the created splits and the teacher's knowledge gets applied. The model code is built in a way where it saves a checkpoint for the current epoch its on and also a best model version. In case accuracy stops increasing and the model plateaus, it has a 7 epoch patience built in, which once it reaches it automatically stops and saves the best model version.
↓
Testing the currently created model with the test_predict.py file before actually pushing the model to the frontend portion of the code
↓
backend.py where everything gets connected to the front end portion. All the prdicting functions are stored in there.


---- FILE INFORMATION AND REFERENCES ----

[Python constants](https://realpython.com/python-constants/#:~:text=Because%20Python%20constants%20are%20just,constants%20use%20uppercase%20letters%20only.)

---- audio-processing file ----
A lot of the initial audio processing code is pretty similar to the previous version of the project, but it does go on to a deeper level of processing data compared to before.

The original 2 datasets used fior the initial 260 classes when creating the project are:
[Xeno-Canto Bird Recordings Extended (A-M)](https://www.kaggle.com/datasets/rohanrao/xeno-canto-bird-recordings-extended-a-m)
[Xeno-Canto Bird Recordings Extended (N-Z)](https://www.kaggle.com/datasets/rohanrao/xeno-canto-bird-recordings-extended-n-z)
Later on the projects lifecycle, the dataset will be expanded with more species

Long term idea for the project is not to limit myself to only birds from the Kaggle dataset, so to be able to have a wide access to all kinds of bird taxonomy info ill be using the [eBird Taxonomy v2025 file](https://www.birds.cornell.edu/clementschecklist/introduction/updateindex/october-2025/2025-citation-checklist-downloads/) to extract information for the corresponding bird class. That way the original Kaglle code names match the scientific ones.
For the corresponding bird classes to be able to add photos/location ill be using the iNaturalist endpoint so i can access the photos, but that will come later on.

Deciding on the amount of used cores for preprocessing - [How the "Number of Workers" Parameter in PyTorch DataLoader Actually Works](https://www.geeksforgeeks.org/deep-learning/how-the-number-of-workers-parameter-in-pytorch-dataloader-actually-works/)

res_type(soxr_hq is chosen) used alongside with the librosa library used for the resampling: [librosa.resample](https://librosa.org/doc/main/generated/librosa.resample.html). Its high quality (second highest behind VHQ, and is also much faster for the resampling as we are handling 500+ classes).

For the console progress when training, processing audios, etc, [Tqdm](https://www.datacamp.com/tutorial/tqdm-python) will be used for the progress bars, as its accurate and lightweight:
To be able to match the .wav and .mp3 types of files, [Globs](https://gulpjs.com/docs/en/getting-started/explaining-globs/) will be used in the audio process classes function.


---- data set splitting file ----
[The Importance of Splitting Datasets into Training, Validation, and Test Sets](https://ruveydakardelcetin.medium.com/the-importance-of-splitting-datasets-into-training-validation-and-test-sets-417caaeae91d)
Creating a seperate data split file. The old project split the data in the exact same file as it was training (the splitting itself took place right before the training) and that took a lot of processing time even tho splitting the data into seperate sets is a one time thing when training a model. Having it happen before the model training allows the training to go faster in terms of speed and also use much less memory.
The data splitting function now also shows the files that each split has, aswell as which classes have too little samples, that way while testing it i can know going forward which classes will need expanding/shrinking.


---- SpecAugment file ----
[text](https://sh-tsang.medium.com/brief-review-specaugment-a-simple-data-augmentation-method-for-automatic-speech-recognition-1ceddfe24e2d)
[text](https://www.researchgate.net/figure/Data-augmentation-specAugment_fig3_382027607)
[text](https://docs.pytorch.org/audio/stable/tutorials/audio_feature_augmentation_tutorial.html)
[text](https://research.google/blog/specaugment-a-new-data-augmentation-method-for-automatic-speech-recognition/)
[text](https://www.semanticscholar.org/paper/SpecAugment%3A-A-Simple-Data-Augmentation-Method-for-Park-Chan/b0fae9fbb4e580d92395eabafe73e317ae6510e3)

---- data resizing file ----
A seperate file that resizes the three seperate sets of data before parsing them to the model training. aAs im using EfficientNetB0, it expects files with a 224x224 size, so resizing the spectrograms doesnt create a loss of information/data and also helps the EfficientNet training be done to the standards.


---- data parser file ----
The data parser idea is to pass only one class at a time while training the model. That way its lighter and easier to keep track of the seperate steps of the model creating process


---- BirdNet teacher file ----
Whole idea is to use "distillation". It involves using the already pre-trained and robust BirdNet model, making it go through my dataset and using the data it has learned to help the main model learn a bit better. It is a measure that deals with accuracy and helps close the gap between validation and train accuracy. if the teacher is trained right, it can substantially improve the model im building.


---- model training file references ----
[choosing between resnet vs efficientnet](https://medium.com/@enrico.randellini/image-classification-resnet-vs-efficientnet-vs-efficientnet-v2-vs-compact-convolutional-c205838bbf49)
Ultimately going for EfficientNet as its a bit better and ultimately much lighter to use
[How to Convert a TensorFlow Model to PyTorch?](https://www.geeksforgeeks.org/deep-learning/how-to-convert-a-tensorflow-model-to-pytorch/)
[How to use GPU acceleration in PyTorch?](https://www.geeksforgeeks.org/deep-learning/how-to-use-gpu-acceleration-in-pytorch/)
[How I changed CPU to GPU support for my ML Model (Easy Guide)](https://medium.com/@vsquarevaibhavverma/how-i-enabled-gpu-support-for-my-ml-model-easy-guide-ready-f5f455358d6d)
[Fine-Tuning BirdNET on Custom Data: Tailoring AI for Local Bird Monitoring](https://medium.com/@guneet.mutreja/fine-tuning-birdnet-on-custom-data-tailoring-ai-for-local-bird-monitoring-0282fe7eaa80)

Main meat of the project. Everything so far comes down to this point in the project. The main model is being created with EfficientNetB0.


---- add samples file ----
A dedicated file who's purpose is to extract more classes/more samples for a chosen class. It uses the Xeno Canto v3 API and a dedicated custom API key.
It has a limit of downloads per request, a maximum set recordings based on how many i want to add to a class, and it only downloads recordings with a A or B rating.


---- backend file references ----
For the connection between the front and backend i will be using 
[FastAPI](https://fastapi.tiangolo.com/)
[First Steps FastApi](https://fastapi.tiangolo.com/tutorial/first-steps/)
[FastAPI for AI: Build an AI Endpoint in 30 Minutes](https://www.youtube.com/watch?v=uDUfZyNXFX0)
[text](https://medium.com/@Dev_sammie/integrating-machine-learning-models-into-frontend-applications-36e849ec1e7f) 
[How to redirect the user back to the home page using FastAPI, after submitting an HTML form?](https://stackoverflow.com/questions/70690454/how-to-redirect-the-user-back-to-the-home-page-using-fastapi-after-submitting-a)