# Implementation-of-Deep-Neural-networks-for-Image-Segregation-and-Priority-Detection
# To use Classifier.ipynb (To train the model):
Load Drive:
from google.colab import drive
drive.mount( '/content/drive/' )
Enter Dataset Directory:
%cd ./AML_Project/Images/Training/
Run all cells sequentially.
To access the trained model:
1. Enter /content/drive/My Drive/AML_Project/Images/Training.
2. Load vgg16_1.h5. The model gets stored in /content/drive/My
Drive/AML_Project/Images/ vgg16_1.h5.
To test the model:
Load the model from: /content/drive/My Drive/AML_Project/Images/ vgg16_1.h5.
1. Enter /content/drive/My Drive/AML_Project/Test.
2. Specify the range of number of images that contain meme, text and human faces. The
ranges can be manually seen( and separated) in :
/content/drive/My Drive/AML_Project/Test/
3. Results of the test sets are stored in the following directories:
● /content/drive/My Drive/AML_Project/Test/Humans_Classified.
● /content/drive/My Drive/AML_Project/Test/Memes_Classified.
● /content/drive/My Drive/AML_Project/Test/Text_Classified.
# To use Face.ipynb (To train the model):
Load Drive:
from google.colab import drive
drive.mount( '/content/drive/' )
Run all cells sequentially.
The model takes input images from the
/content/drive/My Drive/AML_Project/Test/Test1_Classified
As an output we have 2 directories
1. Priority images
2. Other images



