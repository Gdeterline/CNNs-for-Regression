import numpy as np
import matplotlib.pyplot as plt
import cv2

def generate_arrow_dataset(n_samples=2000, img_size=64, angle_range=(-90, 90)):
    X = np.zeros((n_samples, img_size, img_size, 1), dtype=np.float32)
    y = np.zeros((n_samples,), dtype=np.float32)
    
    center = (img_size // 2, img_size // 2)
    length = img_size // 3

    for i in range(n_samples):
        angle = np.random.uniform(*angle_range)
        y[i] = angle
        
        # image vide
        img = np.zeros((img_size, img_size), dtype=np.uint8)
        
        # point de départ et d'arrivée de la flèche
        end_x = int(center[0] + length * np.sin(np.deg2rad(angle)))
        end_y = int(center[1] - length * np.cos(np.deg2rad(angle)))

        cv2.arrowedLine(img, center, (end_x, end_y), 255, 2, tipLength=0.3)
        X[i, :, :, 0] = img / 255.0  # normalisation

    return X, y


def test_generate_arrow_dataset(n_samples=6):
    X, y = generate_arrow_dataset(n_samples=n_samples)
    fig, axes = plt.subplots(1, n_samples, figsize=(12, 2))
    for i, ax in enumerate(axes):
        ax.imshow(X[i].squeeze(), cmap='gray')
        ax.set_title(f"{y[i]:.1f}°")
        ax.axis('off')
    plt.show()


if __name__ == "__main__":
    test_generate_arrow_dataset(4)