from .data_utils import load_images
from .vis_utils import visualize_first32_eigen_faces, visualize_image_reconstruction, visualize_interleaved_eigen_faces, \
    visualize_pc_importance, visualize_reconstruction_mse
from src.models.pca import PCA
from src.settings import IMG_HEIGHT, IMG_WIDTH

pca_method = "svd"

if __name__ == '__main__':
    img_array = load_images() / 255.

    pca = PCA()

    pca.fit(img_array, pca_method)
    pc_ls, var_ls = pca.pc_ls, pca.var_ls

    # Eigen Face Visualization
    visualize_first32_eigen_faces(pc_ls, IMG_HEIGHT, IMG_WIDTH)
    visualize_interleaved_eigen_faces(pc_ls, IMG_HEIGHT, IMG_WIDTH)

    # Eigen Face Importance
    visualize_pc_importance(var_ls, k_pcs=100)

    # Face Reconstruction Visualization
    visualize_image_reconstruction(img_array, pc_ls, IMG_HEIGHT, IMG_WIDTH)
    visualize_reconstruction_mse(img_array, pc_ls, IMG_HEIGHT, IMG_WIDTH)
