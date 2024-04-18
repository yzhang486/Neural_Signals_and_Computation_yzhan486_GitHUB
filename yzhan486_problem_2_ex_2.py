## Part B
from scipy.stats import skew
##  I would like to try standard deviation and skewness

std_dev_img = np.std(frames, axis=0)
skewness_img = skew(frames, axis=0, bias=False)

# Visualization using matplotlib
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Displaying the Standard Deviation image
axes[0].imshow(std_dev_img, cmap='gray')
axes[0].set_title('Standard Deviation Image')
axes[0].axis('off')  # Hide axis ticks and labels

# Displaying the Skewness image
axes[1].imshow(skewness_img, cmap='gray')
axes[1].set_title('Skewness Image')
axes[1].axis('off')  # Hide axis ticks and labels

plt.tight_layout()
plt.show()