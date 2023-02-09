import numpy as np
import matplotlib.colors

# Enviroatlas definitions
enviroatlas_labels = [0, 10, 20, 30, 40, 52, 70, 80, 82, 91, 92]
enviroatlas_label_descriptions = {
    0:  "Unclassified",
    10: "Water",
    20: "Impervious Surface",
    30: "Soil and Barren",
    40: "Trees and Forest",
    52: "Shrubs",
    70: "Grass and Herbaceous",
    80: "Agriculture",
    82: "Orchards",
    91: "Woody Wetlands",
    92: "Emergent Wetlands"
}
enviroatlas_label_mapping = {
    0:  0,
    10: 1,
    20: 2,
    30: 3,
    40: 4,
    52: 5,
    70: 6,
    80: 7,
    82: 8,
    91: 9,
    92: 10,
}
enviroatlas_label_mapping_array = np.array([enviroatlas_label_mapping[i] if i in enviroatlas_label_mapping.keys() else 0 
                                            for i in range(max(enviroatlas_labels))])
enviroatlas_cmap = matplotlib.colors.ListedColormap(
    [
        [1.00000000, 1.00000000, 1.00000000],
        [0.00000000, 0.77254902, 1.00000000],
        [0.61176471, 0.61176471, 0.61176471],
        [1.00000000, 0.66666667, 0.00000000],
        [0.14901961, 0.45098039, 0.00000000],
        [0.80000000, 0.72156863, 0.47450980],
        [0.63921569, 1.00000000, 0.45098039],
        [0.86274510, 0.85098039, 0.22352941],
        [0.67058824, 0.42352941, 0.15686275],
        [0.72156863, 0.85098039, 0.92156863],
        [0.42352941, 0.62352941, 0.72156863],
    ]
)
enviroatlas_label_colors = {
    0:  (255*1.00000000, 255*1.00000000, 255*1.00000000),
    1:  (255*0.00000000, 255*0.77254902, 255*1.00000000),
    2:  (255*0.61176471, 255*0.61176471, 255*0.61176471),
    3:  (255*1.00000000, 255*0.66666667, 255*0.00000000),
    4:  (255*0.14901961, 255*0.45098039, 255*0.00000000),
    5:  (255*0.80000000, 255*0.72156863, 255*0.47450980),
    6:  (255*0.63921569, 255*1.00000000, 255*0.45098039),
    7:  (255*0.86274510, 255*0.85098039, 255*0.22352941),
    8:  (255*0.67058824, 255*0.42352941, 255*0.15686275),
    9:  (255*0.72156863, 255*0.85098039, 255*0.92156863),
    10: (255*0.42352941, 255*0.62352941, 255*0.72156863),
}

# Simplified Enviroatlas definitions
enviroatlas_simplified_labels = [0, 1, 2, 3, 4, 5]
enviroatlas_simplified_label_descriptions = {
    0: 'Unclassified',
    1: 'Water',
    2: 'Impervious surface',
    3: 'Soil and barren',
    4: 'Trees and forest (and shrub in AZ)',
    5: 'Grass and herbaceous',
}
enviroatlas_simplified_label_mapping = {
    0:  0,
    10: 1,
    20: 2,
    30: 3,
    40: 4,
    52: 4,
    70: 5,
    80: 5,
    82: 5,
    91: 4,
    92: 4,
}
enviroatlas_simplified_label_mapping_array = np.array([enviroatlas_simplified_label_mapping[i] if i in enviroatlas_simplified_label_mapping.keys() else 0 
                                                       for i in range(max(enviroatlas_labels))])
enviroatlas_simplified_cmap = matplotlib.colors.ListedColormap(
    [
        [1.00000000, 1.00000000, 1.00000000],
        [0.00000000, 0.77254902, 1.00000000],
        [0.61176471, 0.61176471, 0.61176471],
        [1.00000000, 0.66666667, 0.00000000],
        [0.14901961, 0.45098039, 0.00000000],
        [0.63921569, 1.00000000, 0.45098039],
    ]
)
enviroatlas_simplified_label_colors = {
    0:  (255*1.00000000, 255*1.00000000, 255*1.00000000),
    1:  (255*0.00000000, 255*0.77254902, 255*1.00000000),
    2:  (255*0.61176471, 255*0.61176471, 255*0.61176471),
    3:  (255*1.00000000, 255*0.66666667, 255*0.00000000),
    4:  (255*0.14901961, 255*0.45098039, 255*0.00000000),
    5:  (255*0.63921569, 255*1.00000000, 255*0.45098039),
}

def labels_to_color(labels, categorical, label_idxs, label_colors):
    '''
    Transforms labels image to color image. 
    
    Args:
        labels (numpy array): Categorical/one-hot labels image (1,height,width)/(n_classes,height,width)
        categorical (bool): Whether the labels image is categorical or one-hot encoded
        label_idxs (list): Integers to use as labels
        label_colors (dict): Colors corresponding to integer labels
    Returns:
        out_img (np.array): Color image corresponding to predicted classes (3,height,width)
    '''
    # Create zeros image and add class colors sequentially
    out_img = np.zeros((3,) + labels.shape[1:])
    
    # Fill with label colors
    if categorical:
        # Save time by adding in colors only of existing labels
        label_idxs = np.unique(labels)
        
        for c in label_idxs:
            color = np.array(label_colors[c])[:, np.newaxis, np.newaxis]
            label_locs = (labels == c).squeeze()
            out_img[0, label_locs] = color[0]
            out_img[1, label_locs] = color[1]
            out_img[2, label_locs] = color[2]
            
        # Cast to int and transpose to (height,width,channels)
        out_img = out_img.astype(int).transpose([1,2,0])
    else:
        for c in label_idxs:
            color = np.array(label_colors[c])[:, np.newaxis, np.newaxis]
            out_img += labels[c] * color
        # Cast to int and transpose to (height,width,channels)
        out_img = out_img.transpose([1,2,0]).astype(int)
    
    return out_img