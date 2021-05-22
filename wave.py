import numpy as np
import cv2
import math


w = 1024
h = w
num_points = 17

points = np.random.rand(num_points,3).tolist()

dists = []
max_dist = 0

for p in points:
    x = int(p[0]*w)
    y = int(p[1]*h)
    point = np.array([x,y])
    point = np.interp(point, [0,w], [w*0.4, w*0.6])
    dist = np.zeros((w,h))
    dist = np.linalg.norm(np.indices(dist.shape)-point[:,None,None], axis=0)
    dists.append(dist)
    if np.max(dist) > max_dist:
        max_dist = np.max(dist)

scale_fact = 0.25
max_dist *= scale_fact

num_frames = 300

imgs = []
min_v = np.finfo(float).max
max_v = np.finfo(float).min


for f in range(0, num_frames):
    img = np.zeros((w,h))
    for i,p in enumerate(points):
        dist = dists[i]
        strength = np.interp(dist, [0,max_dist], [1, 0])
        freq = np.interp(p[2], [0,1], [0.1,0.3])
        img += np.sin(dist*freq + f/(num_frames/10)) * strength 
    imgs.append(img)
    if np.max(img) > max_v:
        max_v = np.max(img)
    if np.min(img) < min_v:
        min_v = np.min(img)


video = cv2.VideoWriter('wave.avi', 0, 30.0, (w,h))

for img in imgs:
    img = np.interp(img, [min_v, max_v], [0,255]).astype(np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    video.write(img)


video.release()
