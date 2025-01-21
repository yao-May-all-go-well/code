# code
## Implementation of the Algorithm
- **`single_file_process.ipynb`**: Demonstrates the preprocessing steps for a single file, providing an intuitive example of the signal processing workflow.  
- **`data_process.py`**: Implements batch preprocessing and feature extraction of signals, which is the first step of the experimental process.  
- **`extract_bispectrum.py`**: Focuses on feature extraction from signals, with this method being called in data_process.py for feature extraction.
- **`models`**: Contains the model definitions and related code used in the experiments.  
- **`train.py`** and **`main.py`**: Handle the training logic and overall workflow management, respectively.

## data
The data comes from two publicly available datasets, namely:
https://github.com/yaseen21khan/
https://physionet.org/content/challenge-2016/1.0.0/

## Dependencies and Requirements
The code depends on Python standard libraries `os`, `random`, `time`, and external libraries `numpy`, `librosa`, `scipy`, `torch`, and `matplotlib`.

## Figures
This is the algorithm flowchart.
![image](https://github.com/user-attachments/assets/01509cc2-3a56-4443-9dd2-6b7d7616f6d6)

Feature
![image](https://github.com/user-attachments/assets/5ea69af3-9031-4d20-8006-8586ee69f4d4)

## Citation
The material provided in this repository is made available for reference and educational purposes. When utilizing or referencing this material, please give appropriate credit by mentioning the original creator and the research article:
Paper:Cui, M., Zhao, Z., & Yin, T. Enhanced Heart Sound Classification Using Bispectral Features and Attention-Guided ResCANet. The Visual Computer.

