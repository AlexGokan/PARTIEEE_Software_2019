The preprocessing script is msertest.py

You run it with the following syntax:
python msertest.py input_image_path output_path
  
The results will be placed in a folder called "PARTIEEE_output_images", and inside that will be the subfolder you chose when running the program

Press any key at the end of the program to close all the image windows that appear.  This can be easily removed later, but is kept in for visualization reasons


The current pipeline is:
msertest.py (detect regions of interest) -----> gmm_test.py (detect dominant colors) --------> ocr_v2.m (recover text and color content)
currently there is no false-positive-rejection built in, but the next version will include a preliminary method built into the gmm script based on color similarity


This program was written using Python 3.7.1, and openCV 4.0.0

Not gauranteed to be compatible with other versions of openCV
