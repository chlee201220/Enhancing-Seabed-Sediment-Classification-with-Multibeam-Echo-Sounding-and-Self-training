import pandas as pd
import numpy as np
import numpy.ma as ma
from skimage import data
from scipy.interpolate import griddata
from matplotlib import pyplot as plt
import get_glcm
import time
from PIL import Image
import tifffile
from scipy.signal import convolve2d



# load file
legacy_interpolation = pd.read_csv('C:/Users/user/Desktop/research/multibeam_segmentation_230320/LegacyData_EastSea/legacy_interpolation.csv')

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
def main():
    pass


if __name__ == '__main__':
    
    main()
     
    start = time.time()

    print('---------------0. Parameter Setting-----------------')
    nbit = 64 
    mi, ma = 0, 255 
    slide_window = 7 
    step = [2]
    angle = [0]
    print('-------------------1. Load Data---------------------') 
    img = masked_bs_grid
    img = np.uint8(255.0 * (img - np.nanmin(img))/(np.nanmax(img) - np.nanmin(img))) # normalization
    
    h, w = img.shape
    print('------------------2. Calcu GLCM---------------------')
    glcm = get_glcm.calcu_glcm(img, mi, ma, nbit, slide_window, step, angle)
    print('-----------------3. Calcu Feature-------------------')
    # 
    for i in range(glcm.shape[2]):
        for j in range(glcm.shape[3]):
            glcm_cut = np.zeros((nbit, nbit, h, w), dtype=np.float32)
            glcm_cut = glcm[:, :, i, j, :, :]
            mean = get_glcm.calcu_glcm_mean(glcm_cut, nbit)
            variance = get_glcm.calcu_glcm_variance(glcm_cut, nbit)
            homogeneity = get_glcm.calcu_glcm_homogeneity(glcm_cut, nbit)
            contrast = get_glcm.calcu_glcm_contrast(glcm_cut, nbit)
            dissimilarity = get_glcm.calcu_glcm_dissimilarity(glcm_cut, nbit)
            entropy = get_glcm.calcu_glcm_entropy(glcm_cut, nbit)
            energy = get_glcm.calcu_glcm_energy(glcm_cut, nbit)
            correlation = get_glcm.calcu_glcm_correlation(glcm_cut, nbit)
            Auto_correlation = get_glcm.calcu_glcm_Auto_correlation(glcm_cut, nbit)
    print('---------------4. Display and Result----------------')
    plt.figure(figsize=(10, 4.5))
    font = {'family' : 'Times New Roman',
    'weight' : 'normal',
    'size'   : 12,
    }
    plt.subplot(2,5,1)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.axis('off')
    plt.imshow(img, cmap ='gray')
    plt.title('Original', font)

    plt.subplot(2,5,2)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.axis('off')
    plt.imshow(mean, cmap ='gray')
    plt.title('Mean', font)

    plt.subplot(2,5,3)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.axis('off')
    plt.imshow(variance, cmap ='gray')
    plt.title('Variance', font)

    plt.subplot(2,5,4)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.axis('off')
    plt.imshow(homogeneity, cmap ='gray')
    plt.title('Homogeneity', font)

    plt.subplot(2,5,5)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.axis('off')
    plt.imshow(contrast, cmap ='gray')
    plt.title('Contrast', font)

    plt.subplot(2,5,6)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.axis('off')
    plt.imshow(dissimilarity, cmap ='gray')
    plt.title('Dissimilarity', font)

    plt.subplot(2,5,7)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.axis('off')
    plt.imshow(entropy, cmap ='gray')
    plt.title('Entropy', font)

    plt.subplot(2,5,8)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.axis('off')
    plt.imshow(energy, cmap ='gray')
    plt.title('Energy', font)

    plt.subplot(2,5,9)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.axis('off')
    plt.imshow(correlation, cmap ='gray')
    plt.title('Correlation', font)

    plt.subplot(2,5,10)
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.axis('off')
    plt.imshow(Auto_correlation, cmap ='gray')
    plt.title('Auto Correlation', font)

    plt.tight_layout(pad=0.5)
    plt.savefig('./GLCM_Features_BS.png'
                , format='png'
                , bbox_inches = 'tight'
                , pad_inches = 0
                , dpi=300)
    plt.show()

    end = time.time()
    print('Code run time:', end - start)



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
    interpolated_maps = {}

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
        interpolated_maps[filter_name] = interpolated_map

    return interpolated_maps

filters = define_laws_filters()
texture_maps = apply_laws_and_interpolate(masked_bs_grid, x_unique, y_unique, filters)

