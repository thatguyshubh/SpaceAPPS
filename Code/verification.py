from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
directory = input("Enter the directory path: ")
data_dir = Path(directory)


b4path = data_dir / "B04RAW.tiff"
b8path = data_dir / "B08RAW.tiff"
b3path = data_dir / "B03RAW.tiff"
#This one is red ig
with rasterio.open(b4path) as r_src:
    b4 = r_src.read(1).astype(np.float32)
    
#this one is near infrared
with rasterio.open(b8path) as nir_src:
    b8 = nir_src.read(1).astype(np.float32)
    
#this one is green
with rasterio.open(b3path) as g_src:
    b3 = g_src.read(1).astype(np.float32)

nir = b8
red = b4
green = b3

denom = red + nir

ndvi = np.where(denom == 0, 0, (nir - red) / np.where(denom == 0, 1.0, denom))


print(f"Bands loaded: red={red.shape}, nir={nir.shape}, green={green.shape}")
print(f"NDVI shape: {ndvi.shape}")
print(f"NDVI min: {ndvi.min():.4f}, max: {ndvi.max():.4f}, mean: {ndvi.mean():.4f}")
print(f"Array shape: {red.shape}")


plt.figure(figsize=(8, 6))
plt.imshow(ndvi, cmap='RdYlGn', vmin=-1, vmax=1)
plt.colorbar(label='NDVI')
plt.title('NDVI Map [Green=vegitation, redness=volcanic rock]')
plt.show()
