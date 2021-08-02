# Practical-Super-Resolution-with-CNFlow

This repository provides the training and testing code for the paper **Bridging the Domain Gap in Real World 
Super-Resolution**.

## Training

#### CNFlow
The first step of the training is to train the CNFlow model. CNFlow model requires a dataset of paired images
at several iso. The dataset of noisy/clean pairs must be organized as follow:

```
|- dataset  |-clean     |- aaa.png
                        |- bbb.png
                        |-...

            |-200       |- aaa.png
                        |- bbb.png
                        |-...
            
            |-iso value |- aaa.png
                        |- bbb.png
                        |-...
            
            |-...
```
To train CNFlow model, run the following command:
``` python train_cnflow.py -opt options/train_cnflow.json```

#### SR model

Our SR model need motion blur and downsampling kernels to be trained, ours can be found [here](https://www.google.com).
Once CNFlow is trained, you must specify CNFlow and the kernels path in _options/train_sr.json_.
Finally, run the following command to train the SR model:
``` python train_sr.py -opt options/train_sr.json```

## Testing
To run our SR model on your dataset please run:
``` python test_sr.py --dataroot PATH_TO_DATA --model_path PATH_TO_MODEL --iso DENOISE_INTENSITY``` 
## Acknowledgments
Our code is inspired by [KAIR](https://github.com/cszn/KAIR).
