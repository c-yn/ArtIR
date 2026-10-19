### Training 
 1. Download the training and test dataset
 2. To train ArtIR on the UHD-Blur dataset, run
```
cd ArtIR
CONFIG=UHD/Options/UHD-Blur.yml python train_ArtIR.py
 ```


### Evaluation
1. Download the pre-trained model and place it in ./pretrained_models/
2. Download the testset, then run
```
python evaluation_from_data.py --data UHD-Blur --test_y_channel

``` 
to generate resulting images
3. Run 
```
python calculate_psnr_ssim.py --data UHD-Blur
``` 
to obtain the numerical results
