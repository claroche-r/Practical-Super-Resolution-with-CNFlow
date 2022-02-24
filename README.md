# Practical-Super-Resolution-with-CNFlow

# Data Generation pipeline

![](images/DataGenPipe.png) 

# Architecture
![](images/PipelineFlow.png) 

# Figures

## SR on GoPro images

HR | Bicubic            |  ESRGAN | IKC | RealSRMD 
:-:|:------------------:|:-------:|:---:|:------:
![](images/gopro/Rect.png)  |  ![](images/gopro/Bicubic.png)  | ![](images/gopro/ESRGAN.png)  | ![](images/gopro/IKC.png)  | ![](images/gopro/RealSRMD.png)  |

## SR on SIDD images

HR | Bicubic            |  ESRGAN | IKC | RealSRMD 
:-:|:------------------:|:-------:|:---:|:------:
![](images/SIDD/rect.png)  |  ![](images/SIDD/Bicubic.png)  | ![](images/SIDD/ESRGAN.png)  | ![](images/SIDD/IKC.png)  | ![](images/SIDD/RealSRMD.png)  |

## Noise generation comparison
Clean | Noisy            |  CNFlow
:----:|:----------------:|:-------:
![](images/Noiseflow/gt_big.png)  |  ![](images/Noiseflow/noisy_big.png)   | ![](images/Noiseflow/noiseflow_big.png) |

## Realworld degradations

 Bicubic            |  ESRGAN | IKC | RealSRMD 
:------------------:|:-------:|:---:|:------:
![](images/AppendixRes/Bicubic.png)  | ![](images/AppendixRes/ESRGAN.png)  | ![](images/AppendixRes/IKC.png)  | ![](images/AppendixRes/RealSRMD.png)  |

## L1 loss vs GAN loss


L1    |  GAN
:------------------:|:-------:
![](images/AppendixRes/PSNR.png)  | ![](images/AppendixRes/GAN.png)  | 



