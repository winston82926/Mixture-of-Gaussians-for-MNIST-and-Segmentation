# Mixture of Gaussians for MNIST and Segmentation
I built GMMs(Gaussian Mixture Models) and tested the performance by MNIST data. Constructing 10 GMMs for 10 digits, and used them to do recognition for the test data. After analyzing the GMMs by tuning parameters, I used it to implement the image segmentation.



## MNIST

|<img src="MNIST/E_k.png" width="80%">|
|:--------------------------------------------:|
|Classification error rate for different value of K|


|<img src="MNIST/mu_sigma.png" width="80%">|
|:--------------------------------------------:|
|Visualization of $\mu_k, \Sigma_k$ for each GMM(digit)|


## Segmentation

<img src="Segmentation/Data/221272.jpg" width=200/>|<img src="Segmentation/Data/221272.jpg_myGmm_k=10.png" width=200/>|<img src="Segmentation/Data/CB_221272.png"/>
:----------------------------------------:|:----------------------------------------:|:----------------------------------------:
<img src="Segmentation/Data/test.jpg" width=200/>|<img src="Segmentation/Data/test.jpg_myGmm_k=10.png" width=200/>|<img src="Segmentation/Data/CB_test.png"/>
Original|K=10|Colormap ($\mu_k$)
