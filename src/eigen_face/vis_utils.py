import matplotlib.pyplot as plt
import numpy as np


def visualize_first32_eigen_faces(pc_ls, img_height, img_width):
    rows, cols = 4, 8
    fig, axs = plt.subplots(rows, cols, figsize=(14, 9))

    plt.suptitle("Eigen Faces", fontsize=22)

    for i in range(rows):
        for j in range(cols):
            pc_idx = i * cols + j

            img = pc_ls[pc_idx].reshape(img_height, img_width)

            axs[i][j].imshow(img, cmap="gray")
            axs[i][j].set_xticks([])
            axs[i][j].set_yticks([])
            axs[i][j].set_xlabel(pc_idx, fontsize=12)

    plt.show()


def visualize_interleaved_eigen_faces(pc_ls, img_height, img_width):
    PC_IDX = [10, 100, 200, 500, 1000, 2000, 5000, 10000]
    rows, cols = 1, len(PC_IDX)

    fig, axs = plt.subplots(rows, cols, figsize=(14, 9))

    for i in range(len(PC_IDX)):
        pc_idx = PC_IDX[i]

        img = pc_ls[pc_idx].reshape(img_height, img_width)

        axs[i].imshow(img, cmap="gray")
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i].set_xlabel(pc_idx, fontsize=12)


def visualize_pc_importance(var_ls, k_pcs=100):
    var_capture = var_ls / var_ls.sum()
    var_cum = var_capture.cumsum()

    plt.figure(figsize=(15, 10))
    plt.stem(var_capture[:k_pcs] * 100)
    plt.ylabel("variance capture %", fontsize=20)
    plt.xlabel("PC (sorted - decreasing variance)", fontsize=20)
    plt.show()

    plt.figure(figsize=(15, 10))
    plt.stem(var_cum[:k_pcs] * 100)
    plt.ylabel("cumulative variance capture %", fontsize=20)
    plt.xlabel("PC (sorted - decreasing variance)", fontsize=20)
    plt.show()


def visualize_image_reconstruction(img_array, pc_ls, IMG_HEIGHT, IMG_WIDTH):
    img_idxs = [50, 200, 250, 100]
    K_PC_SELECT_LS = [50, 100, 1000, 5000, 10000]

    img_ls = img_array[img_idxs]
    img_recons = []

    for idx, k_pc_select in enumerate(K_PC_SELECT_LS):
        pc_selected_ls = pc_ls[:k_pc_select]  # K x P
        proj_arr = img_ls @ pc_selected_ls.T  # N x K
        img_recon = proj_arr @ pc_selected_ls  # N x P

        img_recons.append(img_recon)

    fig, axs = plt.subplots(len(img_idxs), len(K_PC_SELECT_LS) + 1, figsize=(10, 7.7))
    for row in range(len(img_idxs)):
        axs[row][0].imshow(img_ls[row].reshape(IMG_HEIGHT, IMG_WIDTH), cmap="gray")
        axs[row][0].set_xticks([])
        axs[row][0].set_yticks([])

        for idx in range(len(K_PC_SELECT_LS)):
            axs[row][idx + 1].imshow(img_recons[idx][row].reshape(IMG_HEIGHT, IMG_WIDTH), cmap="gray")
            axs[row][idx + 1].set_xticks([])
            axs[row][idx + 1].set_yticks([])
            if row == len(img_idxs) - 1:
                axs[row][idx + 1].set_xlabel(f"K = {K_PC_SELECT_LS[idx]}")
    axs[row][0].set_xlabel(f"Original Image")
    plt.show()


def visualize_reconstruction_mse(img_array, pc_ls, IMG_HEIGHT, IMG_WIDTH):
    pc_select_range_ls = [np.arange(50, IMG_HEIGHT * IMG_WIDTH, 100), np.arange(1, 101)]

    for pc_select_range in pc_select_range_ls:
        mse_ls = []
        count_pcs = []
        for k_pc_select in pc_select_range:
            pc_selected_ls = pc_ls[:k_pc_select]  # K x P
            proj_arr = img_array @ pc_selected_ls.T  # N x K
            img_recon = proj_arr @ pc_selected_ls  # N x P
            mse = ((img_array - img_recon) ** 2).mean()

            mse_ls.append(mse)
            count_pcs.append(k_pc_select)

        plt.figure(figsize=(20, 10))
        plt.stem(count_pcs, mse_ls)
        plt.xticks(count_pcs, rotation=90)
        plt.ylabel("MSE(original img, reconstructed img)", fontsize=20)
        plt.xlabel("#PCs", fontsize=20)
        plt.show()
