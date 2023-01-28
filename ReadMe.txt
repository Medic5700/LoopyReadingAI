Loopy Reading AI
Author: Medic5700

The project is an AI that can transcribe text from a picture, but structuring the AI in such a way as to have the flaws of the human vision system.
Being made as part of Devember2022, and subsicuencly derailed by a motherboard failure.

Development Stack:
    Windows 10
    Windows Terminal
    Python 3.10
    pip3
        pillow
        numpy
        torch
            Microsoft Visual C++ Redistributable
        torchvision

Verify Development Stack:
    import torch
    x = torch.rand(5, 3)
    print(x)

    # The above should output soemthing similar to the following
    tensor([[0.4175, 0.6417, 0.4817],
        [0.5940, 0.6668, 0.9841],
        [0.2784, 0.5837, 0.7389],
        [0.7290, 0.5148, 0.7714],
        [0.6131, 0.4797, 0.4553]])
