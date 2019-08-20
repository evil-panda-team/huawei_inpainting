# Test
* Testing script is ./pipeline.py (the same as ./run.py)
* --img_dir argument need to be specified (path to the directory, containing images)

The test script can work both on gpu and cpu:
* To test on GPU: tester = Tester(model, input_size, batch_size)
* To test on CPU: tester = Tester(model, input_size, batch_size, 'cpu')

# Weights
Download 'model_places2.pth' weights from here: https://drive.google.com/open?id=1SGJ_Z9kpchdnZ3Qwwf4HnN-Cq-AeK7vH and place it inside ./DFNet/model/ directory

# Train
* Training script is ./DFNet/train_vlad.py, is based on this: https://github.com/Yukariin/DFNet/blob/master/train.py
* The model was trained on Places365-Standard high-resolution images: http://places2.csail.mit.edu/download.html
* To train the model you need to create training and validaton .flist files, containing the paths to images
* Example of how to generate your own .flist files is in ./DFNet/flist.py script

# Dependencies
* python3
* OpenCV
* PyTorch
* numpy
* tqdm
* tensorboardX (for training only)