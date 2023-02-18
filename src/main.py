import glob
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from scipy.signal import find_peaks
from skimage.restoration import denoise_wavelet
from itertools import pairwise
from fpdf import FPDF


def scipy_find_peaks(data):
    local_maxima_x, _ = find_peaks(data)
    local_minima_x, _ = find_peaks(-data)
    peaks_x = np.sort(np.concatenate((local_maxima_x, local_minima_x), axis=0))
    return peaks_x


def normalized_distance(x1, y1, x2, y2, min_x, max_x, min_y, max_y):
    dx = (x2 - x1) / (max_x - min_x)
    dy = (y2 - y1) / (max_y - min_y)
    return math.sqrt(dx ** 2 + dy ** 2)


def peak_filter_algorithm(data, peaks_x, k):
    peaks_y = [data[x] for x in peaks_x]
    min_x, max_x = 0, len(data)
    min_y, max_y = np.min(data), np.max(data)
    filtered_peaks = set(peaks_x)
    for (x1, y1), (x2, y2) in pairwise(zip(peaks_x, peaks_y)):
        distance = normalized_distance(x1, y1, x2, y2, min_x, max_x, min_y, max_y)
        if distance <= k and x1 in filtered_peaks and x2 in filtered_peaks:
            filtered_peaks.remove(x1)
            filtered_peaks.remove(x2)
    filtered_peaks = list(filtered_peaks)
    filtered_peaks.sort()
    return filtered_peaks


def create_graph(ax, title, data, data_denoise, x, y):
    ax.set_title(f"{title} Peaks Graph")
    ax.set_xlabel("Video Frame (#)")
    ax.set_ylabel("Calcium Intensity")
    ax.plot(data, label="Original", zorder=0)
    ax.plot(data_denoise, label="Wavelet Denoise", zorder=1)
    ax.scatter(x, y, label="Peak", color="yellow", zorder=2)
    ax.legend()


def create_table(ax, title, x, y):
    ax.set_title(f"{title} Peaks Locations ({len(x)})")
    ax.table(list(zip(x, y)), colLabels=('X', 'Y'), loc='center', cellLoc='center')
    ax.axis('off')


def create_row(index, axs, title, data, data_denoise, x, y):
    create_graph(axs[index + 1][0], title, data, data_denoise, x, y)
    create_table(axs[index + 1][1], title, x, y)


def generate_graphs(file_name, k_values):
    i = 1
    cell_data = pd.read_csv(file_name)
    while True:
        column_name = f"Mean{i}"
        try:
            data = cell_data[column_name]
        except KeyError:
            break

        print(f"Processing Cell {i}")
        data_denoise = denoise_wavelet(data,
                                       method='VisuShrink',
                                       mode='soft',
                                       wavelet_levels=4,
                                       wavelet='sym8',
                                       rescale_sigma='True')

        fig, axs = plt.subplots(nrows=1 + len(k_values), ncols=2, figsize=(12.8, 4.8 * (1 + len(k_values))))
        fig.suptitle(f"Cell {i}", fontsize=16)

        all_peaks_x = scipy_find_peaks(data_denoise)
        all_peaks_y = [data_denoise[x] for x in all_peaks_x]
        create_graph(axs[0][0], "All", data, data_denoise, all_peaks_x, all_peaks_y)
        create_table(axs[0][1], "All", all_peaks_x, all_peaks_y)

        for index, k in enumerate(k_values):
            title = f"Filter (k={k})"
            filter_peaks_x = peak_filter_algorithm(data_denoise, all_peaks_x, k)
            filter_peaks_y = [data_denoise[x] for x in filter_peaks_x]
            create_row(index, axs, title, data, data_denoise, filter_peaks_x, filter_peaks_y)

        file_name = f"../output/Cell-{i}.png"
        fig.savefig(file_name)
        plt.close(fig)
        i += 1
    print(i)
    return i


def pack_to_pdf(file, total_cells):
    pdf = FPDF()
    pdf.add_page()
    for i in range(1, total_cells):
        image = f"../output/Cell-{i}.png"
        print(f"Packing {image}")
        pdf.image(image, w=200)
    pdf.output(file, "F")
    pdf.close()


def remove_image_files():
    for f in glob.glob("../output/*.png"):
        os.remove(f)


def main(input_file, k_values, output_pdf_file=None, keep_image_files=False):
    print("Generating graphs...")
    total_cells = generate_graphs(input_file, k_values)

    if output_pdf_file:
        print(f"Packing to PDF - {output_pdf_file}...", )
        pack_to_pdf(output_pdf_file, total_cells)

    if not keep_image_files:
        print("Deleting image files...")
        remove_image_files()

    print("Done")


if __name__ == "__main__":
    main(input_file="../example/graph_data.csv",
         k_values=[0.01, 0.02, 0.04],
         output_pdf_file="../output/results.pdf")
