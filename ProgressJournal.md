General progress for the app and its development, seperate from the backend journal (that one is more for me to track react progress).
Will contain the sources for the datasets used aswell as different references for the model training, articles etc.


Day 1: Gathering dataset and reading on about the project. SMall setups which file will be where.
Mostly planning.

Day 2: Quite a bit of research and preplanning done, starting work on the integration of the audio files into the system and their functions.
Work on the functions where the audio files will be cut up into 5s segments and then run through a few cheks (if they're long enough, active enough, etc).

Day 3: The audio processing function is complete, and for now it seems to work good. It saves all the different npy files for their respective classes, making saving expanding the data set easy. Will see how it progresses once it gets to the model training if any adjustments to the audio extraction need to be done

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

Day 5: Working on the model training portion of the code

Day 6: