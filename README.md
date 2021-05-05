# DoCN
Script for predicting bounding box over page and Image Document Scanner

### Instructions to use  
Launch terminal and run the following commands  
``` 
    git clone https://github.com/kahanikaar/DoCN.git
    cd DoCN
    pip install -r requirements.txt
    python main.py [Input Image Path]
```
All the Output Images are stored in DoCN/Outputs/ directory.  
  
## Libraries using
``` 
    rembg==1.0.16
    opencv-python==4.2.0.34
    imutils==0.5.3
    numpy==1.19.4
    matplotlib==3.2.1
    pillow>=8.0.1
    
```
All libaries to be installed from PyPi.  
  
  
  
Script inspired from [docscan](https://github.com/danielgatis/docscan) maintained by [Daniel Gatis](https://github.com/danielgatis)
