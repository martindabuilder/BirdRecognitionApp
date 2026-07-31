File containing all the references and progress on the creation of the backend portion of the project.

---- audio-processing file references ----
A lot of the initial audio processing code is pretty similar (if not identical) to the previous version of the project, so that eliviates quite a bit of work at the starting stages atleast, most of the tryharding will likely come with the model training since itll be with more species this time

The original 2 datasets used fior the initial 260 classes when creating the project are:
[Xeno-Canto Bird Recordings Extended (A-M)](https://www.kaggle.com/datasets/rohanrao/xeno-canto-bird-recordings-extended-a-m)
[Xeno-Canto Bird Recordings Extended (N-Z)](https://www.kaggle.com/datasets/rohanrao/xeno-canto-bird-recordings-extended-n-z)
Later on the projects lifecycle, the dataset will be expanded with more species

Considering the N-Z collection's CSV file is wrong from the Kaggle download, ill be using the eBird taxonomy CSV file for the corresponding bird class to the code from the xeno canto  [eBird Taxonomy v2025](https://www.birds.cornell.edu/clementschecklist/introduction/updateindex/october-2025/2025-citation-checklist-downloads/)

For the corresponding bird classes to be able to add photos/location ill be using the iNaturalist endpoint so i can access the photos, but that will come later on

Deciding on the amount of used cores for preprocessing - [How the "Number of Workers" Parameter in PyTorch DataLoader Actually Works](https://www.geeksforgeeks.org/deep-learning/how-the-number-of-workers-parameter-in-pytorch-dataloader-actually-works/)

res_type(soxr_hq is chosen) used with the librosa library used for the resampling [librosa.resample](https://librosa.org/doc/main/generated/librosa.resample.html). Its high quality ( second highest behind VHQ, and is also much faster for the resampling as we are handling 500+ classes)

For the console progress when training, processing audios etc Tqdm will be used for the progress bars, as its accurate and lightweight: [Tqdm Python: A Guide With Practical Examples](https://www.datacamp.com/tutorial/tqdm-python)

To be able to match the .wav and .mp3 types of files glob will be used in the process classes function [Explainig Globs](https://gulpjs.com/docs/en/getting-started/explaining-globs/)

---- model training file references ----

---- backend file references ----