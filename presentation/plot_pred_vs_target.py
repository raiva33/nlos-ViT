import matplotlib.pyplot as plt
import os
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage import io, img_as_float

# Define the paths to your folders
predictions_folder = '/home/mao/Documents/code/mitsuba2-transient-nlos/presentation/prediction/translation'
ground_truth_folder = '/home/mao/Documents/code/mitsuba2-transient-nlos/presentation/target/translation'

# Get the filenames in the folders
prediction_files = sorted(os.listdir(predictions_folder))
ground_truth_files = sorted(os.listdir(ground_truth_folder))

print(prediction_files, ground_truth_files)

# Create a figure to display the images
fig, axes = plt.subplots(4, 2, figsize=(10, 10))
fig.tight_layout()

# Loop through the images
for i, (pred_file, gt_file) in enumerate(zip(prediction_files, ground_truth_files)):
    # Read the images
    pred_image = img_as_float(io.imread(os.path.join(predictions_folder, pred_file)))
    gt_image = img_as_float(io.imread(os.path.join(ground_truth_folder, gt_file)))

    # Check the filenames match
    assert pred_file[:15] == gt_file[:15], "Filenames must match"

    # Calculate SSIM and PSNR
    similarity_index, _ = ssim(pred_image, gt_image, full=True)
    peak_snr = psnr(pred_image, gt_image)

    # Display the predicted image
    axes[i, 0].imshow(pred_image, cmap='gray')
    axes[i, 0].set_title(f'Prediction\nSSIM: {similarity_index:.2f}, PSNR: {peak_snr:.2f}')
    axes[i, 0].axis('off')

    # Display the ground truth image
    axes[i, 1].imshow(gt_image, cmap='gray')
    axes[i, 1].set_title('Ground Truth')
    axes[i, 1].axis('off')

plt.show()
