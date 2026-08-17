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


Day 7: Reworking the data parser, aswell as adding a new file: resize_dataset. In the training test i did, the training itself was a bit slower since each epoch needed time to resize all the spectrograms to 160x160 size to better fit the EfficientNet model requirements. however that added more stress on my laptop each epoch, aswell as computing time, each epoch going from ~1.30 minutes to a bit over 2 minutes. So now with the new file what ends up happening is once the dataset is split into the 3 new sets, then each set gets resized into 160x160 after running the resize file, and the model trains directly on that. A marginal but noticeable improvement in time and computing power needs, saving ~30s to a minute per epoch once the project grows (50+ epochs potentially and about 200+ more classes) will definitely add up in the long run.
Starting to build up a proper model training file now too, and from there on its going to get fine tuned and tested/improved further.


Day 7 + 8 combined since i did work on the 7th and then on the 8th day but the model seems to work, still have yet to test it for how accurate it guesses birds but for now it compiles and also runs normally so thats definitely a good start for now. The model also has real time evaluation during its training, and the training itself is in 2 phases: first phsae with a frozen backbone and then 2nd phase unfreezes it to fine tune it.
Also going to be working on a file that will add more samples to my classes. The plan is to use it with the Xeno Canto API to download more classes of the same bird directly from there. WIll try to add more functionality to it like getting only the higher rated audios etc.
+ day 9 combined because the same work is stretched between the 3 days, but its been mostly tinkering and testing the model. Its accuracy so far has been sadly a bit too low but ill try my best to step it up, might look into trying a different sort of model to see if it yields better results. I have added a function that can easily add more classes/samples to already existing classes only adding up recordings with a rating of A or B. Im trying my best to balance out the classes aswell so that the classes arent completely unbalanced (some classes turned out to have near 20k samples when ran through the audio processing file, meanwhile others only gather 3 or 4). 
The file that adds samples is put in a gitignore since it contains the Xeno Canto API and a custom API Key, so for the sake of privacy i have hidden it for now.

Day 10: After quite a bit of changing around the model learns better and actually classifies birds properly now. It still needs to be fine tuned quite a bit which is what ive been doing today and the dataset has to be balanced a bit more before i start to create a real backend version for it but this is good progress.
In terms of balancing out the dataset im aiming for each class to have atleast close to 1k samples, will take up quite a bit of space on the disk but thats inevitable when trying to build a model from the ground up. 
Im slowly starting to design ideas for the front end which ill be starting work on pretty soon, hopefully i get the model fine tuned and working by the next few days so i can start chipping away at the front end.
Added back all the checks for the spectrograms being too short/too dark in the audio processing functions as i took them out to test a different approach to the model, also adjusted the overlap to be 1s between consecutive segments instead of 2.5.
Removed the random shuffling of the dataset when splitting it into seperate train/test/validation sets. Also added to the audio processing/splitting tracking the file IDs so that they get properly seperated into proper splits. That way a single .npy file doesnt get seperated into 2 seperate splits.


Day 11: Lots of changes, biggest one is trying to make BirdNet be used as a "teacher" model with knowledge distillation. This is done in an attempt to help the main model learn better, so if it works out this will be of huge help tp the model and training it.
Added quite a few new files to the dataset aswell trying to balance out the dataset a bit more, still some work need to be done but its much more balanced now. There are a few classes im considering dropping that have too little samples and xeno canto doesnt have any with better audio quality to add onto the dataset, so the list might go down one or two classes as the training goes on.


Day 12: Spent most of the day downloading more samples and testing out ResNet50 alongside the teacher model. Learning with the teacher gave a slightly better result but its accuracy and validations loss/accuracy is still too low. Certain tests with the model seemed to go well but its still not sure at all with the classifications, so the coming days will be trying to get its accuracy as up as best as i could, and slowly start to build up the front end too.


Day 13: Switched back from ResNet to EfficientNet as the better results ResNet gave were mostly because i balanced out the dataset before testing it. The dataset is now even more balanced so EfficientNet on a new run gave substantially better results. Got rid of the distillation from the teacher aswell for this test run and the validation accuracy peaked at 51.91% which is a pretty big improvement over the 40 to barely 42% it gave before. The model seems to be getting more confident in its results aswell which is an improvement, even tho i still want it to classify given test audios with atleast 50% certainty before i start shifting focus to the frontend.
A big part of the model doing better is because the spectrograms are now processed with 32k sample rate, rather than the 16k one before, that gives them a lot more detail and makes it easier for the model to learn their patterns.
The model training itself is a bit more balanced now as it has power balancing included in it, that way classes with lower sample size dont get swallowed by bigger ones.
The balancing of the dataset gave pretty substantial results so ill look into the lower sample sized classes and consider removing most classes under 500 samples as i cant expand them in any way sadly as xeno canto has no more recordings for them, but that will depend on if the model performs again better tomorrow with the hopefully added teacher back into the training mix.


Day 14: Trying to bring back the teacher distillation approach again but this time with a more robust teacher that is better trained. First test run for the teacher is with a higher files per class count now with 20 for each class (it was 5 before) and a bigger top predicted classes (20 now, was 10 before).