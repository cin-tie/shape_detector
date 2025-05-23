import numpy as np
import cv2

SHAPES = ['circle', 'square', 'triangle', 'star', 'hexagon', 'pentagon']
COLORS = {
    'red': (255, 0, 0),
    'green': (0, 255, 0),
    'blue': (0, 0, 255)
}

def draw_shape(img, shape, color):
    h, w, _ = img.shape
    center = (w//2, h//2)
    size = min(h, w) // 3

    if shape == 'circle':
        cv2.circle(img, center, size, color, -1)
    elif shape == 'square':
        top_left = (center[0]-size, center[1]-size)
        bottom_right = (center[0]+size, center[1]+size)
        cv2.rectangle(img, top_left, bottom_right, color, -1)
    elif shape == 'triangle':
        pts = np.array([
            [center[0], center[1]-size],
            [center[0]-size, center[1]+size],
            [center[0]+size, center[1]+size]
        ], np.int32)
        cv2.fillPoly(img, [pts], color)
    elif shape == 'star':
        # Пример простой звезды (5-лучевая)
        pts = np.array([
            [center[0], center[1]-size],
            [center[0]+int(size*0.35), center[1]-int(size*0.35)],
            [center[0]+size, center[1]-int(size*0.35)],
            [center[0]+int(size*0.5), center[1]+int(size*0.15)],
            [center[0]+int(size*0.7), center[1]+size],
            [center[0], center[1]+int(size*0.5)],
            [center[0]-int(size*0.7), center[1]+size],
            [center[0]-int(size*0.5), center[1]+int(size*0.15)],
            [center[0]-size, center[1]-int(size*0.35)],
            [center[0]-int(size*0.35), center[1]-int(size*0.35)]
        ], np.int32)
        cv2.fillPoly(img, [pts], color)
    elif shape == 'hexagon':
        pts = np.array([
            [center[0]-size, center[1]],
            [center[0]-size//2, center[1]-int(size*0.87)],
            [center[0]+size//2, center[1]-int(size*0.87)],
            [center[0]+size, center[1]],
            [center[0]+size//2, center[1]+int(size*0.87)],
            [center[0]-size//2, center[1]+int(size*0.87)]
        ], np.int32)
        cv2.fillPoly(img, [pts], color)
    elif shape == 'pentagon':
        pts = np.array([
            [center[0], center[1]-size],
            [center[0]+int(size*0.95), center[1]-int(size*0.31)],
            [center[0]+int(size*0.59), center[1]+int(size*0.81)],
            [center[0]-int(size*0.59), center[1]+int(size*0.81)],
            [center[0]-int(size*0.95), center[1]-int(size*0.31)]
        ], np.int32)
        cv2.fillPoly(img, [pts], color)
    return img

def generate_dataset(num_samples=1000, img_size=64):
    X = []
    y_shape = []
    y_color = []
    for _ in range(num_samples):
        img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255  # белый фон
        shape_idx = np.random.randint(0, len(SHAPES))
        color_idx = np.random.randint(0, len(COLORS))
        shape = SHAPES[shape_idx]
        color_name = list(COLORS.keys())[color_idx]
        color = COLORS[color_name]

        img = draw_shape(img, shape, color)

        X.append(img)
        y_shape.append(shape_idx)
        y_color.append(color_idx)

    return np.array(X), np.array(y_shape), np.array(y_color)
