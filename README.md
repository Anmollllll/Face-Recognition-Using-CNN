# Face-Recognition-Using-CNN
This project implements a face recognition system using Convolutional Neural Networks (CNNs) without a frontend interface. It is structured with multiple Python files, each handling different aspects of the pipeline.
Project Structure:
enter.py - Asks for the Name and ID of the student and automatically saves the photo of the student.A csv and excel file is created.Excel to store the details and csv file is to be used for futher purposes
imgcrop.py - crops the captured faces with face only
greyscale.py - converts the cropped images to black and white
trainmodel.py - trains the CNN model with the greyscale images
detectface.py - detects the face and marks attendance if face is recognized
