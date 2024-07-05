import pandas as pd
import numpy as np
import numpy.ma as ma
from scipy.interpolate import griddata
import cv2
import pywt
from scipy.signal import convolve2d
from skimage.feature import graycomatrix, graycoprops



# load file
legacy_interpolation = pd.read_csv('legacy_interpolation.csv')

x = legacy_interpolation['x'].values
y = legacy_interpolation['y'].values
bs = legacy_interpolation['bs'].values
bathy = legacy_interpolation['bathy'].values

x_unique = np.unique(x)
y_unique = np.unique(y)
X, Y = np.meshgrid(x_unique, y_unique)

bs_grid = np.zeros_like(X)
bathy_grid = np.zeros_like(X)

for i, x_val in enumerate(x_unique):
    for j, y_val in enumerate(y_unique):
        mask = (x == x_val) & (y == y_val)
        bs_grid[j, i] = np.mean(bs[mask])  
        bathy_grid[j, i] = np.mean(bathy[mask])

mask = np.isnan(bs_grid)
masked_bs_grid = ma.array(bs_grid, mask=mask)



# GLCM feature extraction
def calc_glcm_features(img, slide_window=7, vmin=0, vmax=255, nbit=64, step=[2], angle=[0]):
    h, w = img.shape
    bins = np.linspace(vmin, vmax+1, nbit+1)
    img1 = np.digitize(img, bins) - 1
    img1 = cv2.copyMakeBorder(img1, slide_window//2, slide_window//2, slide_window//2, slide_window//2, cv2.BORDER_REPLICATE)

    patches = np.zeros((slide_window, slide_window, h, w), dtype=np.uint8)
    for i in range(h):
        for j in range(w):
            patches[:, :, i, j] = img1[i: i + slide_window, j: j + slide_window]

    features = {}
    for i in range(h):
        for j in range(w):
            glcm = graycomatrix(patches[:, :, i, j], step, angle, levels=nbit)
            features[(i, j)] = {
                'mean': calc_feature(glcm, lambda p, q: p / nbit**2),
                'variance': calc_feature(glcm, lambda p, q: (p - features[(i, j)]['mean'])**2),
                'homogeneity': calc_feature(glcm, lambda p, q: 1 / (1. + (p - q)**2)),
                'contrast': calc_feature(glcm, lambda p, q: (p - q)**2),
                'dissimilarity': calc_feature(glcm, lambda p, q: np.abs(p - q)),
                'entropy': calc_feature(glcm, lambda p, q: -np.log10(p + 1e-10)),
                'energy': calc_feature(glcm, lambda p, q: p**2)
            }

    return features

def calc_feature(glcm, func):
    nbit = glcm.shape[0]
    result = np.zeros((glcm.shape[2], glcm.shape[3]), dtype=np.float32)
    for i in range(nbit):
        for j in range(nbit):
            result += func(i, j) * glcm[i, j]
    return result

if __name__ == '__main__':
    img = masked_bs_grid  
    img = np.uint8(255 * (img - np.nanmin(img)) / (np.nanmax(img) - np.nanmin(img)))
    glcm_features = calc_glcm_features(img)



# DWT feature extraction
def extract_dwt_features(data, x_coords, y_coords, levels=3, wavelet='haar'):
    current_data = data.copy()
    interpolated_results = {}   
    x_original, y_original = np.meshgrid(x_coords, y_coords)
    
    for level in range(1, levels + 1):
        coeffs = pywt.dwt2(current_data.filled(fill_value=np.nan), wavelet)
        LL, (LH, HL, HH) = coeffs
        
        sub_bands = {'LL': LL, 'LH': LH, 'HL': HL, 'HH': HH}
        for name, sub_band_data in sub_bands.items():
            x_dwt, y_dwt = np.meshgrid(np.linspace(np.min(x_coords), np.max(x_coords), sub_band_data.shape[1]),
                                       np.linspace(np.min(y_coords), np.max(y_coords), sub_band_data.shape[0]))
            
            sub_band_flat = sub_band_data.ravel()
            points_dwt = np.array([x_dwt.ravel(), y_dwt.ravel()]).T
            
            interpolated_data = griddata(points_dwt, sub_band_flat, (x_original, y_original), method='nearest')
            
            interpolated_results[f'{name}{level}'] = interpolated_data
        
        current_data = ma.array(LL, mask=np.isnan(LL))
    return interpolated_results

dwt_features = extract_dwt_features(masked_bs_grid, x_unique, y_unique)



# Law's texture feature extraction
def define_laws_filters():
    L5 = np.array([1, 4, 6, 4, 1]).reshape(1, 5)
    E5 = np.array([-1, -2, 0, 2, 1]).reshape(1, 5)
    S5 = np.array([-1, 0, 2, 0, -1]).reshape(1, 5)
    W5 = np.array([-1, 2, 0, -2, 1]).reshape(1, 5)
    R5 = np.array([1, -4, 6, -4, 1]).reshape(1, 5)

    filters = {
        'L5E5': convolve2d(L5.T, E5, mode='full'),
        'E5L5': convolve2d(E5.T, L5, mode='full'),
        'E5S5': convolve2d(E5.T, S5, mode='full'),
        'S5E5': convolve2d(S5.T, E5, mode='full'),
        'S5R5': convolve2d(S5.T, R5, mode='full'),
        'W5R5': convolve2d(W5.T, R5, mode='full'),
        'R5W5': convolve2d(R5.T, W5, mode='full'),
        'L5S5': convolve2d(L5.T, S5, mode='full'),
        'S5L5': convolve2d(S5.T, L5, mode='full')
    }
    return filters

def apply_laws_and_interpolate(data, original_x, original_y, filters, padding_mode='symmetric'):
    padded_data = np.pad(data.filled(fill_value=np.nan), 2, mode=padding_mode)
    laws_features = {}

    for filter_name, filter_array in filters.items():
        texture_map = convolve2d(padded_data, filter_array, mode='valid')
        new_x = np.linspace(np.min(original_x), np.max(original_x), texture_map.shape[1])
        new_y = np.linspace(np.min(original_y), np.max(original_y), texture_map.shape[0])
        
        interpolated_map = griddata(
            np.array([new_x.ravel(), new_y.ravel()]).T,
            texture_map.ravel(),
            (np.meshgrid(original_x, original_y)),
            method='nearest'
        )
        laws_features[filter_name] = interpolated_map

    return laws_features

filters = define_laws_filters()
laws_features = apply_laws_and_interpolate(masked_bs_grid, x_unique, y_unique, filters)

