# PII_Detection
Installation Tutorial
* To download super-gradients:
run on wsl, make sure you're up to date on the python library that works on ubuntu - important!
Usually if you are not, you'll get a warning message in the terminal.
then run: 
pip install super-gradients
* How to run the current version 31.5.24:
/bin/python3 /mnt/c/Users/noyro/Documents/PII_Detection/YOLO-NAS/train.py
On this directory:
/mnt/c/Users/noyro/Documents/PII_Detection

* When it finishes, modify the model path in YOLO-nas file based on the latest run.
you get this from this path:
\checkpoints\yolo_nas_s_experiment\RUN_20240531_150808_855290
and get the average_model.pth file.

* Then you run inteference file.