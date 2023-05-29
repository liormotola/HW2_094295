import torch
from torchvision import transforms
from PIL import Image


# Load the image
image = Image.open("/home/student/hw2/hw2_094295/data/test/i/ab4e0182-ce5d-11eb-b317-38f9d35ea60f.png")  # Replace with the actual path to your image

noisy_image = transforms.GaussianBlur(kernel_size=(5,9), sigma=(0.1, 5))(image)
# # Convert the image to a PyTorch tensor
# image_tensor = TF.to_tensor(image)
#
# # Generate random Gaussian noise with the same size as the image tensor
# noise_tensor = torch.randn_like(image_tensor)
#
# # Define the noise intensity
# noise_intensity = 0.2  # Adjust this value to control the intensity of the noise
#
# # Add the noise to the image tensor
# noisy_image_tensor = image_tensor + noise_intensity * noise_tensor
#
# # Clip the pixel values to the valid range [0, 1]
# noisy_image_tensor = torch.clamp(noisy_image_tensor, 0.0, 1.0)
# noisy_image = TF.to_pil_image(noisy_image_tensor)
# Save the noisy image
noisy_image.save("/home/student/hw2/hw2_094295/adjusted_image3.png")

