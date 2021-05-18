import numpy as np
import cv2
import math

def dist(p1,p2):
    return np.linalg.norm(p1-p2)

w = 1024
h = w
num_points = 17

img = np.zeros((w,h))
points = np.random.rand(num_points,3).tolist()

dists = []
max_dist = 0

for p in points:
    x = int(p[0]*w)
    y = int(p[1]*h)
    point = np.array([x,y])
    point = np.interp(point, [0,w], [w*0.4, w*0.6])
    dist = np.linalg.norm(np.indices(img.shape)-point[:,None,None], axis=0)
    dists.append(dist)
    if np.max(dist) > max_dist:
        max_dist = np.max(dist)

scale_fact = 0.25
max_dist *= scale_fact

for i,p in enumerate(points):
    dist = dists[i]
    strength = np.interp(dist, [0,max_dist], [1, 0])
    phase = np.interp(p[2], [0,1], [0.1,0.3])
    img += np.sin(dist*phase) * strength 
    
img = np.interp(img, [np.min(img), np.max(img)], [0,255])


cv2.imwrite("wave.png", img)