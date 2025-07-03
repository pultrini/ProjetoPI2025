import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import affine_transform, zoom
from PIL import Image, ImageDraw
from scipy.ndimage import center_of_mass

# region: Image Creation
def create_pyramid(image, levels):
    """Creates a pyramid of downsampled images."""
    pyramid = [image]
    for i in range(levels - 1):
        pyramid.append(zoom(pyramid[-1], 0.5))
    return pyramid

def create_phantom_image(size=(256, 256), shape='circle', shape_size=50):
    """Creates a simple phantom image with a specified shape."""
    img = Image.new('L', size, 0)
    draw = ImageDraw.Draw(img)
    if shape == 'circle':
        draw.ellipse([(size[0]/2 - shape_size, size[1]/2 - shape_size),
                      (size[0]/2 + shape_size, size[1]/2 + shape_size)], fill=255)
    elif shape == 'rectangle':
        draw.rectangle([(size[0]/2 - shape_size, size[1]/2 - shape_size),
                        (size[0]/2 + shape_size, size[1]/2 + shape_size)], fill=255)
    return np.array(img, dtype=np.float32) / 255.0

def preprocess_images(fixed_img, moving_img):
    """
    Resizes the moving image to match the dimensions of the fixed image.
    """
    if isinstance(fixed_img, np.ndarray):
        fixed_img_pil = Image.fromarray((fixed_img * 255).astype(np.uint8))
    else:
        fixed_img_pil = fixed_img

    if isinstance(moving_img, np.ndarray):
        moving_img_pil = Image.fromarray((moving_img * 255).astype(np.uint8))
    else:
        moving_img_pil = moving_img

    moving_img_resized = moving_img_pil.resize(fixed_img_pil.size, Image.Resampling.LANCZOS)

    fixed_array = np.array(fixed_img_pil, dtype=np.float32) / 255.0
    moving_array = np.array(moving_img_resized, dtype=np.float32) / 255.0
    
    return fixed_array, moving_array

# endregion

# region: Transform Functions
def similarity_transform(image, params):
    """
    Applies a forward similarity transform (scale, rotation, translation) to an image.
    It works by calculating the inverse transformation and applying it to the
    output coordinates, as required by scipy.ndimage.affine_transform.
    """
    s, theta, tx, ty = params
    theta_rad = np.deg2rad(theta)
    center = np.array(image.shape) / 2.0

    inv_s = 1.0 / s if s != 0 else 1.0
    inv_theta_rad = -theta_rad

    inv_rot_matrix = np.array([
        [np.cos(inv_theta_rad), -np.sin(inv_theta_rad)],
        [np.sin(inv_theta_rad),  np.cos(inv_theta_rad)]
    ])

    matrix = inv_rot_matrix * inv_s

    translation = np.array([ty, tx])
    offset = center - matrix @ center - matrix @ translation

    return affine_transform(
        image,
        matrix,
        offset=offset,
        order=1,
        mode='constant',
        cval=0.0
    )
# endregion

# region: Cost Function
def ssd(image1, image2):
    """Computes the Sum of Squared Differences (SSD) between two images."""
    return np.sum((image1 - image2)**2)
# endregion

# region: Optimizer
def compute_gradient(fixed_image, moving_image, params, h=1e-5):
    """Computes the gradient of the SSD using finite differences."""
    grad = np.zeros_like(params)
    for i in range(len(params)):
        params_plus = np.copy(params)
        params_plus[i] += h
        
        params_minus = np.copy(params)
        params_minus[i] -= h
        
        transformed_plus = similarity_transform(moving_image, params_plus)
        cost_plus = ssd(fixed_image, transformed_plus)
        
        transformed_minus = similarity_transform(moving_image, params_minus)
        cost_minus = ssd(fixed_image, transformed_minus)
        
        grad[i] = (cost_plus - cost_minus) / (2 * h)
    return grad

def adam_optimizer(fixed_image, moving_image, initial_params, learning_rate, num_iterations, trainable_params, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """Performs Adam optimization to find optimal transformation parameters."""
    params = np.copy(initial_params)
    cost_history = []
    m = np.zeros_like(params)
    v = np.zeros_like(params)
    
    for t in range(1, num_iterations + 1):
        transformed_moving = similarity_transform(moving_image, params)
        cost = ssd(fixed_image, transformed_moving)
        cost_history.append(cost)
        
        grad = compute_gradient(fixed_image, moving_image, params)
        
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad**2)
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        
        update = learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        params[trainable_params] -= update[trainable_params]
        
        if t % 10 == 0:
            print(f"Iteration {t}/{num_iterations}, Cost: {cost:.2f}, Params: {params}")
            
    return params, cost_history
# endregion

# region: Main Execution
if __name__ == '__main__':
    fixed_image = Image.open('fixed_image.png').convert('L')
    moving_image = Image.open('moving_image.png').convert('L')
    print("Preprocessing images to match sizes...")
    fixed_image, moving_image = preprocess_images(fixed_image, moving_image)
    print(f"Shapes after preprocessing: Fixed={fixed_image.shape}, Moving={moving_image.shape}")

    num_pyramid_levels = 5
    fixed_pyramid = create_pyramid(fixed_image, num_pyramid_levels)
    moving_pyramid = create_pyramid(moving_image, num_pyramid_levels)

    optimal_params = np.array([1.0, 0.0, 0.0, 0.0])
    cost_history = []

    for level in range(num_pyramid_levels - 1, -1, -1):
        print(f"\n--- Optimizing at Pyramid Level {level} ---")
        fixed_level = fixed_pyramid[level]
        moving_level = moving_pyramid[level]

        scale_factor = 2**level
        current_params = np.copy(optimal_params)
        current_params[2:] /= scale_factor

        learning_rate = 0.1
        num_iterations = 300

        current_params, history = adam_optimizer(
            fixed_level, moving_level, current_params, learning_rate, num_iterations, [True, True, True, True]
        )
        cost_history.extend(history)

        optimal_params = np.copy(current_params)
        optimal_params[2:] *= scale_factor

    corregistred_image = similarity_transform(moving_image, optimal_params)

    plt.figure(figsize=(18, 10))

    plt.subplot(2, 3, 1)
    plt.title("Fixed Image")
    plt.imshow(fixed_image, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.title("Moving Image")
    plt.imshow(moving_image, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.title("Corregistred Image")
    plt.imshow(corregistred_image, cmap='gray')
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.title("Difference (Before)")
    plt.imshow(fixed_image - moving_image, cmap='gray')
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.title("Difference (After)")
    plt.imshow(fixed_image - corregistred_image, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.title("Cost Function Convergence")
    plt.plot(cost_history)
    plt.xlabel("Iteration")
    plt.ylabel("SSD Cost")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig('results.png')
    plt.show()

    print("\n--- Ground Truth vs. Optimized Parameters ---")
    print(f"Optimized Parameters: s={optimal_params[0]:.3f}, theta={optimal_params[1]:.3f}, tx={optimal_params[2]:.3f}, ty={optimal_params[3]:.3f}")
# endregion