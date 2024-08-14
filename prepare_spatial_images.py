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

image_raw = sdata['he_image'].data.compute()

align_matrix = np.genfromtxt(alignment_matrix_path, delimiter=",", dtype=float)

# sc.pp.subsample(sdata['table'], random_state=42, n_obs=100)
adata_RNA = sdata['table']

times, times2, times3 = [], [], []

width = 128
height = 128

align_matrix = np.linalg.inv(align_matrix)
geoms = adata_RNA.obs['cell_id'][:2]
shapes = spatialdata.transform(sdata["cell_circles"], to_coordinate_system="global").loc[geoms, ["geometry", "radius"]]

t0 = time.time()
for geom, shape, radius in zip(geoms, shapes["geometry"], shapes["radius"]):
    t1 = time.time()
    coords_x = shape.x
    coords_y = shape.y
    
    cor_coords = align_matrix @ np.array([coords_x, coords_y, 1])
    coords_y_new, coords_x_new = cor_coords[0], cor_coords[1]

    x_min, x_max = coords_x_new - (width / 2), coords_x_new + (width / 2)
    y_min, y_max = coords_y_new - (height / 2), coords_y_new + (height / 2)
    
    image = image_raw[:, int(x_min): int(x_max), int(y_min): int(y_max)].transpose(1,2,0)
    image = np.rot90(image, 1, axes=(0,1))

    print(image.shape)
    print(np.fromstring(image.tostring(), dtype=int).shape)

    # radius = math.ceil(radius)
    # # mask = np.zeros((width, height))
    # # if radius < width and radius < height:
    # #     mask[int(width/2)-radius: int(width/2)+radius, int(height/2)-radius: int(height/2)+radius] = 256
    # # print(mask)
    
    # arr = np.arange(-int(width/2), int(width/2)) ** 2
    # mask = np.add.outer(arr, arr) < radius ** 2
    # # or: arr[:, None] + arr[None, :] < radius ** 2

    # im = Image.fromarray(mask).convert("L")
    # im.save(f"mask{geom}.png")
    
    # im = Image.fromarray(image, 'RGB')
    # im.save(f"your_file{geom}.jpeg")

print(time.time()-t0)

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
