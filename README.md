# code
## Citation
The material provided in this repository is made available for reference and educational purposes. When utilizing or referencing this material, please give appropriate credit by mentioning the original creator and the research article:
Paper:Cui, M., Zhao, Z., & Yin, T. Enhanced Heart Sound Classification Using Bispectral Features and Attention-Guided ResCANet. The Visual Computer.


## Implementation of the Algorithm
- **`single_file_process.ipynb`**: Demonstrates the preprocessing steps for a single file, providing an intuitive example of the signal processing workflow.  
- **`data_process.py`**: Implements batch preprocessing and feature extraction of signals, which is the first step of the experimental process.This is the preprocessing for a five classification dataset.
- **`data_process_two.py`** This is the preprocessing for a binary classification dataset.
- **`extract_bispectrum.py`**: Focuses on feature extraction from signals, with this method being called in data_process.py for feature extraction.
- **`models`**: Contains the model definitions and related code used in the experiments.  
- **`train.py`** and **`main.py`**: Handle the training logic and overall workflow management, respectively.
- **`train_two.py`**: This is the training for a binary classification dataset.

## data
The data comes from two publicly available datasets, namely:
https://github.com/yaseen21khan/
https://physionet.org/content/challenge-2016/1.0.0/

## Dependencies and Requirements
The code depends on Python standard libraries `os`, `random`, `time`, and external libraries `numpy`, `librosa`, `scipy`, `torch`, and `matplotlib`.

## Figures
This is the algorithm flowchart.
![image](https://github.com/user-attachments/assets/01509cc2-3a56-4443-9dd2-6b7d7616f6d6)

Step 1: Preprocessing of the PCG Signal
The preprocessing steps included noise filtering, resampling, and normalization.
Running the script will preprocess the audio and extract features. A screenshot of the feature extraction process is shown below.
![image](https://github.com/user-attachments/assets/5ea69af3-9031-4d20-8006-8586ee69f4d4)
Step 2：Run the train.py file to execute the training code.



