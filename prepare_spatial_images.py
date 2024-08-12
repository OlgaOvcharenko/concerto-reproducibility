import math
import os
import time
import spatialdata
import spatialdata_io
from spatialdata_io.readers.xenium import xenium_aligned_image
import numpy as np
import matplotlib.pyplot as plt
from spatialdata import rasterize
from PIL import Image
import scanpy as sc


data_path="./Multimodal_pretraining/data/data/Xinium/Xenium_V1_humanLung_Cancer_FFPE_outs"
alignment_matrix_path = "./Multimodal_pretraining/data/data/Xinium/Xenium_V1_humanLung_Cancer_FFPE_he_imagealignment.csv"
he_path = "./Multimodal_pretraining/data/data/Xinium/Xenium_V1_humanLung_Cancer_FFPE_he_image.ome.tif"

sdata = spatialdata_io.xenium(data_path)

image_raw = xenium_aligned_image(he_path, alignment_matrix_path)
sdata['he_image'] = image_raw

image_raw = image_raw.data.compute()
print(image_raw.shape)

align_matrix = np.genfromtxt(alignment_matrix_path, delimiter=",", dtype=float)

# sc.pp.subsample(sdata['table'], random_state=42, n_obs=100)
adata_RNA = sdata['table']

times, times2, times3 = [], [], []

width = 128
height = 128

# oinlhbpf-1
# for geom, shape in zip(adata_RNA.obs['cell_id'][:3], spatialdata.transform(sdata["cell_circles"], to_coordinate_system="global").loc[adata_RNA.obs['cell_id'][:3], "geometry"]):
for geom, shape in zip(['oinlhbpf-1'], [spatialdata.transform(sdata["cell_circles"], to_coordinate_system="global").loc["oinlhbpf-1", "geometry"]]):
    t1 = time.time()
    coords_x = shape.x
    coords_y = shape.y
    x_min, x_max = coords_x - (width / 2), coords_x + (width / 2)
    y_min, y_max = coords_y - (height / 2), coords_y + (height / 2)
    times2.append(time.time()-t1)

    t5 = time.time()
    image = image_raw[:, int(x_min): int(x_max), int(y_min): int(y_max)].transpose(1,2,0)
    im = Image.fromarray(image, 'RGB')
    im.save(f"your_file{geom}.jpeg")
    times2.append(time.time()-t5)

    t3 = time.time()
    image = rasterize(
        sdata["he_image"],
        # image_raw,
        ["x", "y"],
        min_coordinate=[x_min, y_min],
        max_coordinate=[x_max, y_max],
        target_unit_to_pixels=1.0,
        target_coordinate_system="global"
    ).data.compute().transpose(1,2,0)
    t2 = time.time()
    times.append(t2-t3)
    times3.append(t2-t1)
    
    im = Image.fromarray(image, 'RGB')
    im.save(f"their_file{geom}.jpeg")

print(np.mean(times2))
print(np.mean(times))
print(np.mean(times3))

# coords_x, coords_y = spatialdata.transform(sdata["cell_boundaries"], to_coordinate_system="global").loc["giphamfc-1", "geometry"].exterior.coords.xy
# x_min, y_min = np.min(coords_x), np.min(coords_y)
# x_max, y_max = np.max(coords_x), np.max(coords_y)

# print(x_min, x_max)
# # print(y_min, y_max)

# slice_width_to_add = width - (x_max - x_min)
# slice_height_to_add = height - (y_max - y_min)

# print(slice_width_to_add)
# # print(slice_height_to_add)

# if slice_width_to_add % 2 == 0:
#     x_min -= slice_width_to_add / 2
#     x_max += slice_width_to_add / 2
#     print(f"1 x")
#     print(x_min, x_max)

# else:
#     x_min -= math.floor(slice_width_to_add / 2)
#     x_max += slice_width_to_add - math.floor(slice_width_to_add / 2)
    
#     print(f"2 x")
#     print(x_min, x_max)

# if slice_height_to_add % 2 == 0:
#     y_min -= slice_height_to_add / 2
#     y_max += slice_height_to_add / 2

# else:
#     y_min -= math.floor(slice_height_to_add / 2)
#     y_max += slice_height_to_add - math.floor(slice_height_to_add / 2)


# if int(x_max - x_min) != 128:
#     x_max += int(x_max - x_min)

#     print(f"3 x")
#     print(x_min, x_max)

# if int(y_max - y_min) != 128:
#     y_max += int(y_max - y_min)

# if int(x_max - x_min) != 128:
#     print(x_min, x_max)

# if int(y_max - y_min) != 128:
#     print(y_min, y_max)

# print(x_min, x_max)
# print(y_min, y_max)

# sdata["rasterized"] = rasterize(
#     sdata["he_image"],
#     ["x", "y"],
#     min_coordinate=[x_min, y_min],
#     max_coordinate=[x_max, y_max],
#     target_unit_to_pixels=1.0,
#     target_coordinate_system="global"
# )

# image = sdata["rasterized"].to_numpy().transpose(1,2,0) #.values

# print(image.shape)

# from PIL import Image
# im = Image.fromarray(image, 'RGB')
# im.save("your_file.jpeg")
