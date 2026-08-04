General progress for the app and its development, seperate from the backend journal (that one is more for me to track react progress).
Will contain the sources for the datasets used aswell as different references for the model training, articles etc.


Day 1: Gathering dataset and reading on about the project. SMall setups which file will be where.
Mostly planning.

Day 2: Quite a bit of research and preplanning done, starting work on the integration of the audio files into the system and their functions.
Work on the functions where the audio files will be cut up into 5s segments and then run through a few checks (if they're long enough, active enough, etc).

Day 3: The audio processing function is complete, and for now it seems to work good. It saves all the different npy files for their respective classes, making saving expanding the data set easy. Will see how it progresses once it gets to the model training and if any adjustments to the audio extraction need to be done.

Day 4: Im building up the structure for the project now, and generally the idea will be:

create spectrograms (audio-processing.py)
↓
splitting the dataset prematurely in train/validation/test sets (dataset_split.py). This way before starting the training i will know which classes also have too few samples to be split up into seperate sets. Making the splitting into a seperate file makes it more memory light as the set splits only need to happen once after all the spectrograms are created.
↓
model training (model_training.py, using functions from data_parser.py to make the workload a bit lighter on my laptop)
↓
testing the model for how good it works using the test_predict.py file before pushing it to the front end
↓
front end model integration (front end not yet started lol)

A bit of a different approach compared to the way i ran the older project, this one is more layered with much more backend archiecturing going on.

Day 5: Working on the model training portion of the code. The code is finished for now and seems to compile without problems, whether or not its fast/slow/reliable will be tested in the coming days of progress.

Day 6: Yesterday i did the model training and the data parsers using tensorflow, and started testing them out today. SOme small bugs and fixes later it turns out it doesnt properly work due to a combination of reasons:y tensorflow is (for better or worse in this case) on a newer version (after version 2.10) so GPU work is not supported, so all the training is done on the CPU, which is already way way slower than what is needed. The previous version of the project was done with tensorflow then but that time it worked due to the difference in smaller amount of classes (only ~15 classes with at most 1.5k spectrograms in total) and the way i used to save the .npy spectrogram files. Big difference this time around is the way im saving the .npy files and accessing it. Before i was making a single big npy file with all the data, now its seperate .npy files for each class containing its respective spectrograms, and in order to save some memory and computing power im making a seperate parser class whos going to have functions that help with parsing the .npy files in a consistent organized way in order to help save memory and gradually load the data set, rather than everything at once.
So a big problem was tensorflow not having GPU support, and due to that it was incredly slow when parsing the classes one by one, taking up to 15+ minutes on an epoch (on a reduced data set for testing purposes, ysing about ~30 classes to test rather than all 260).
And a deicison was made to redo those 2 entire files and make the model training with PyTorch. Many sources and small tests show that it recognizes my GPU and manages to use it, so that seems like the best option going forward rather than trying to run different setups just to get tensorflow to work and loose time.
In testing so far atleast it seems that the dataset_split and audio_processing files work good and do their job reliably and fast regardless of the dataset size (more data means a bit more time obviously but theyre fast enough for a single system project).