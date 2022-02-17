# Style Transfer

The program performs a style transfer from one image to another. 
Input consists of two images: style image (with a style that we want to transfer) and content image (change the style of).

![style_image](assets/st_img.jpg)
![content_image](assets/cnt_img.jpg) 
![output_image](assets/output.jpg)

It is an implementation of the method proposed by Gatys et al. 
Deep learning framework selected for this project is PyTorch. 
If CUDA is available on your device, the program will automatically switch to GPU.

Example usage:
```
python3 style_transfer.py --content-path content.jpg --style-path style.jpg --output-path out.jpg

```

The training loop goes through 1000 iterations, but you can change that with the _n-iterations_ parameter. 
You can also speed up the entire process by decreasing image size with the _size_ parameter.
