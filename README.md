In Neural Style Transfer, we used a pretrained VGG19 model that generates a new image that preserves the structure of the content image while applying textures and colors from the style image.



Problem Statement: To develop a Computer Vision system that can transform ordinary images into artistic representations by combining content and style features.



Working:

1\. It uses pretrained VGG19 model

2\. It then extracts:

&#x20;  i). Content features (structure)

&#x20;  ii). Style features (texture, color)

3\. Optimizes a new image using:

&#x20;  i). Content Loss

&#x20;  ii). Style Loss



Technologies Used

1\. Python

2\. PyTorch

3\. OpenCV / PIL

4\. Deep Learning (CNN)



How to Run:

1\. Install dependencies: pip install -r requirements.txt

2\. Run the script: python style\_transfer.py

3\. Provide:

&#x20;  i). Content image path

&#x20;  ii). Style image path

&#x20;  iii). Output file name



Output: 

The generated image is saved automatically in: C:\\Users\\asus\\Documents\\computer\_vision\_project\\output

