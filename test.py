from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def rayleigh_energy_theorem_image(image): 

    energy_time_domain = np.sum(np.abs(image) ** 2)
     
    fft_image = np.fft.fft2(image) 
    energy_frequency_domain = np.sum(np.abs(fft_image) ** 2) / (image.shape[0] * image.shape[1])
    
    return energy_time_domain, energy_frequency_domain
 
image = Image.open('example_image.png').convert('L')   
image_array = np.array(image) / 255.0   

# Rayleigh Enerji Teoremi'ni uygulama
energy_time, energy_freq = rayleigh_energy_theorem_image(image_array)

print("Zaman etki alanındaki enerji:", energy_time)
print("Frekans etki alanındaki enerji:", energy_freq)

# Resmi gösterme
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image_array, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(np.log(np.abs(np.fft.fftshift(np.fft.fft2(image_array)))), cmap='gray')
plt.title('Frequency Domain')
plt.axis('off')

plt.show()
