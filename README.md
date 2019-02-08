# Style Transfer

![content_image](assets/cnt_img.jpg) 
![style_image](assets/st_img.jpg)
![output_image](assets/output.jpg)

In this repository I present a program that performs a style transfer. 
Given a content image and a style image, the program attempts to transfer style of the style image onto the content image.
This is an implementation of the method proposed by Gatys et al. and uses PyTorch. 
If CUDA is available on your device then the program will automatically switch to GPU, otherwise all computation will be performed on CPU.
By default the training loop goes through 1000 iterations, but you can change that with _n-iterations_ parameter.
You can also speed up the entire process by decreasing image size with _size_ parameter.

Example usage:
```
python3 style_transfer.py --content-path content.jpg --style-path style.jpg --output-path out.jpg

```

I also include Google Colab copy of my development notebook in nb directory.