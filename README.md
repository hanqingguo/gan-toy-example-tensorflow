# Toy example of GAN 

Link for the original paper: https://arxiv.org/abs/1406.2661

Inspired by the blog: 
http://blog.aylien.com/introduction-generative-adversarial-networks-code-tensorflow/, 
wanted to reproduce this toy example. 

To train the model:
    
    $ python main.py
    
The below picture is the result after training 100000 steps, blue curve is the generator's data output G(z),
where z is sampled from uniform distribution. The green curve is the real data distribution and it is sampled 
from gaussian distribution with mean 3.0, standard deviation 3.0.
![Alt text](/pic/GAN.png?raw=true "Trained toy example")


# TODO:

Try this on MNIST dataset.