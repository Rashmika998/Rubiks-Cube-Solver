import numpy as np
import cv2
from tqdm import trange
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(); sns.set_style('dark')
from ipywidgets import interact, interactive, fixed, interact_manual

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression

def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

def gray(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def uint8ify(img):
    img -= img.min()
    img /= img.max()
    img *= 255
    return np.uint8(img)

def overlay(a,b):
    # a and b should be float images and between 0-1

    mask = a >= 0.5 # generate boolean mask of everywhere a > 0.5 
    ab = np.zeros_like(a) # generate an output container for the blended image 

    # now do the blending 
    ab[~mask] = (2*a*b)[~mask] # 2ab everywhere a<0.5
    ab[mask] = (1-2*(1-a)*(1-b))[mask] # else this
    
    return ab

def disp(img, title='', s=12, vmin=None, vmax=None, write=False, file_name=None):
    plt.figure(figsize=(s,s))
    plt.axis('off')
    if vmin is not None and vmax is not None:
        plt.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
    else:
        plt.imshow(img, cmap='gray')
    plt.title(title)
    if write and file_name is not None:
        plt.savefig(file_name)
    plt.show()
    
def before_after(img_a, img_b, name="", vmin=None, vmax=None, effect_name="Processed"):
    fig, axs = plt.subplots(1,2, constrained_layout=True, figsize=(10,4))
    axs[0].axis('off')
    axs[1].axis('off')
    axs[0].set_title(f"{name} Original")
    axs[1].set_title(f"{name} {effect_name}")
    if vmin is not None and vmax is not None:
        axs[0].imshow(img_a, cmap='gray', vmin=vmin, vmax=vmax)
        axs[1].imshow(img_b, cmap='gray', vmin=vmin, vmax=vmax)
    else:
        axs[0].imshow(img_a, cmap='gray')
        axs[1].imshow(img_b, cmap='gray')
    plt.show()

# img_3 = rgb(cv2.imread("input/rubiks1_nologo_sm.jpg"))
# img_3 = rgb(cv2.imread("rubiks2.png"))
img_3 = rgb(cv2.imread("rubiks3.jpg"))
# img_3 = rgb(cv2.imread("input/rubiks5.jpg"))
# img_5 = rgb(cv2.imread("input/rubiks5.jpg"))

# img_3_gray = cv2.imread("input/rubiks1_nologo_sm.jpg", 0) #to make a gray image, 0 is passed at the end. for more https://www.geeksforgeeks.org/python-opencv-cv2-imread-method/
# img_3_gray = cv2.imread("rubiks2.png", 0)
img_3_gray = cv2.imread("rubiks3.jpg", 0)
# img_3_gray = cv2.imread("input/rubiks5.jpg", 0)
# img_5_gray = cv2.imread("input/rubiks5.jpg", 0)
disp(img_3)
img_3 = cv2.fastNlMeansDenoisingColored(img_3,None,10,10,7,21)
disp(img_3)

def shadow_remove(img):
    rgb_planes = cv2.split(img)
    result_norm_planes = []
    for plane in rgb_planes:
        dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = 255 - cv2.absdiff(plane, bg_img)
        norm_img = cv2.normalize(diff_img,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        result_norm_planes.append(norm_img)
    shadowremov = cv2.merge(result_norm_planes)
    return shadowremov

# sh=shadow_remove(img_3_gray)
# disp(sh)

kernel_size = 7
# img_gray = cv2.cvtColor(img_3, cv2.COLOR_BGR2GRAY)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))
img_3_gray_denoise = cv2.fastNlMeansDenoising(img_3_gray,None,20,7,21)

gray = cv2.morphologyEx(img_3_gray_denoise, cv2.MORPH_OPEN, kernel)
# disp(gray)

blur_gray = cv2.GaussianBlur(gray,(kernel_size, kernel_size),0)
disp(blur_gray)


# img_gray = cv2.GaussianBlur(img_gray,(kernel_size, kernel_size),0)
# disp(img_gray)
# # se=cv2.getStructuringElement(cv2.MORPH_RECT , (7,7))
# # bg=cv2.morphologyEx(img_gray, cv2.MORPH_DILATE, se)
# # out_gray=cv2.divide(img_gray, bg, scale=255)
# # disp(out_gray)
# # out_gray = cv2.GaussianBlur(out_gray,(kernel_size, kernel_size),0)
# # disp(out_gray)

divide = shadow_remove(blur_gray)
disp(divide)
# blur_gray = cv2.fastNlMeansDenoising(gray,None,20,7,21)
divide = cv2.fastNlMeansDenoising(divide,None,10,7,21)
# disp(blur_gray)


@interact(low_threshold=(0, 200, 1), high_threshold=(0, 200, 1))
def f(low_threshold = 160, high_threshold = 180):
    edges = cv2.Canny(divide, low_threshold, high_threshold, apertureSize = 5, 
                 L2gradient = True)
    disp(edges)



edges = cv2.Canny(divide, 0, 75,apertureSize = 5, 
                 L2gradient = True) #min threshold is 0, max is 75 where the points above the max considered as edges according to https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html
disp(edges)


rho = 1  # distance resolution in pixels of the Hough grid
theta = np.pi / 180  # angular resolution in radians of the Hough grid
threshold = 15  # minimum number of votes (intersections in Hough grid cell)
min_line_length = 40  # minimum number of pixels making up a line
max_line_gap = 20  # maximum gap in pixels between connectable line segments
line_image = np.copy(img_3) * 0  # creating a blank to draw lines on

# Run Hough on edge detected image
# Output "lines" is an array containing endpoints of detected line segments
lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                    min_line_length, max_line_gap)
print(len(lines))
angles = []
for line in lines:
    for x1,y1,x2,y2 in line:
        angle = cv2.fastAtan2(float(y2-y1), float(x2-x1))
        if angle >= 180:
            angle -= 180
        angles.append(angle)
        # print(angle)
        cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)

# Draw the lines on the  image
img_3_gray_rgb = np.repeat(img_3_gray[:,:,None], 3, 2)
lines_edges = cv2.addWeighted(img_3_gray_rgb, 0.8, line_image, 1, 0)

disp(lines_edges)


angles = np.array(angles)
plt.figure(figsize=(10,6), dpi=100)
plt.scatter(np.arange(len(angles)), angles)
plt.yticks(np.arange(0, 181, step=30))
plt.grid(True)
plt.ylabel("degree")
plt.xlabel("line_id")
plt.show()

plt.figure(figsize=(10,5), dpi=100)
plt.hist(angles.ravel(), bins=100)
plt.xticks(np.arange(0, 181, step=30))
plt.ylabel("frequency")
plt.xlabel("degree")
plt.show()


angle_limits = [0,60,120,180]
rotation_angles = [0, 60, 120]

angles_clustering = KMeans(n_clusters=3, n_init=2)
angles_clustering.fit(angles.reshape(-1,1))
sorted_centers = sorted(angles_clustering.cluster_centers_)

angle_limits[1] = ((sorted_centers[0] + sorted_centers[1]) // 2)[0]
angle_limits[2] = ((sorted_centers[1] + sorted_centers[2]) // 2)[0]

rotation_angles[0] = -(90 - sorted_centers[1])[0] # vertical
rotation_angles[1] = -(90 - sorted_centers[0])[0] # positive
rotation_angles[2] = -(90 - sorted_centers[2])[0] # negative

print("> Cluster centers:")
print([int(center[0]) for center in sorted_centers], end="\n\n")
print("> Breaking angles:")
print(angle_limits, end="\n\n")
print("> Rotation angles:")
print(rotation_angles)


colors = [
    (245, 0, 87), #rgb(245, 0, 87)
    (0, 230, 118), #rgb(0, 230, 118)
    (25, 118, 210), #rgb(25, 118, 210)
    (245, 124, 0), #rgb(245, 124, 0)
    (124, 77, 255) #rgb(124, 77, 255)
]

colors_01 = [
    (245/255, 0/255, 87/255), #rgb(245, 0, 87)
    (0/255, 230/255, 118/255), #rgb(0, 230, 118)
    (25/255, 118/255, 210/255), #rgb(25, 118, 210)
    (245/255, 124/255, 0/255), #rgb(245, 124, 0)
    (124/255, 77/255, 255/255) #rgb(124, 77, 255)
]


rho = 1  # distance resolution in pixels of the Hough grid
theta = np.pi / 180  # angular resolution in radians of the Hough grid
threshold = 15  # minimum number of votes (intersections in Hough grid cell)
min_line_length = 40  # minimum number of pixels making up a line
max_line_gap = 20  # maximum gap in pixels between connectable line segments
line_image = np.copy(img_3) * 0  # creating a blank to draw lines on

# Run Hough on edge detected image
# Output "lines" is an array containing endpoints of detected line segments
lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                    min_line_length, max_line_gap)

angles = []
vertical_lines = []
negative_lines = []
positive_lines = []

for idx, line in enumerate(lines):
    for x1,y1,x2,y2 in line:
        angle = cv2.fastAtan2(float(y2-y1), float(x2-x1))
        if angle>=180:
            angle -= 180
        angles.append(angle)

        if angle_limits[0]<=angle<=angle_limits[1]:
            cluster_id = 0
            positive_lines.append([x1,y1,x2,y2])
        if angle_limits[1]<=angle<=angle_limits[2]:
            cluster_id = 1
            vertical_lines.append([x1,y1,x2,y2])
        if angle_limits[2]<=angle<=angle_limits[3]:
            cluster_id = 2
            negative_lines.append([x1,y1,x2,y2])
            
        cv2.line(line_image,(x1,y1),(x2,y2),colors[cluster_id],5)

# Draw the lines on the  image
img_3_gray_rgb = np.repeat(img_3_gray[:,:,None], 3, 2)
lines_edges = cv2.addWeighted(img_3_gray_rgb, 1, line_image, 1, 0)

disp(lines_edges)


# vertical lines
points1 = np.zeros((len(vertical_lines)*2, 2))
for i in range(len(vertical_lines)):
    # point A on a line
    points1[2*i][0] = (vertical_lines[i][0])
    points1[2*i][1] = (vertical_lines[i][1])
    # point B on a line
    points1[2*i+1][0] = (vertical_lines[i][2])
    points1[2*i+1][1] = (vertical_lines[i][3])

    
plt.figure(figsize=(8,8), dpi=100)
plt.title("Vertical Lines")
plt.scatter(points1[:,0], points1[:,1], c='g')
plt.grid(True)
plt.gca().invert_yaxis()
plt.ylabel("y")
plt.xlabel("x")
plt.show()


# positive lines
points2 = np.zeros((len(positive_lines)*2, 2))
for i in range(len(positive_lines)):
    # point A on a line
    points2[2*i][0] = (positive_lines[i][0])
    points2[2*i][1] = (positive_lines[i][1])
    # point B on a line
    points2[2*i+1][0] = (positive_lines[i][2])
    points2[2*i+1][1] = (positive_lines[i][3])
    
plt.figure(figsize=(8,8), dpi=100)
plt.title("Positive Lines")
plt.scatter(points2[:,0], points2[:,1], c='r')
plt.grid(True)
plt.gca().invert_yaxis()
plt.ylabel("y")
plt.xlabel("x")
plt.show()


# negative lines
points3 = np.zeros((len(negative_lines)*2, 2))
for i in range(len(negative_lines)):
    # point A on a line
    points3[2*i][0] = (negative_lines[i][0])
    points3[2*i][1] = (negative_lines[i][1])
    # point B on a line
    points3[2*i+1][0] = (negative_lines[i][2])
    points3[2*i+1][1] = (negative_lines[i][3])
    
plt.figure(figsize=(8,8), dpi=100)
plt.title("Negative Lines")
plt.scatter(points3[:,0], points3[:,1])
plt.grid(True)
plt.gca().invert_yaxis()
plt.ylabel("y")
plt.xlabel("x")
plt.show()


plt.figure(figsize=(8,8), dpi=100)
plt.title("All of the lines")
plt.scatter(points1[:,0], points1[:,1],c='g', s=80, alpha=0.75)
plt.scatter(points2[:,0], points2[:,1],c='r', s=80, alpha=0.75)
plt.scatter(points3[:,0], points3[:,1],c='b', s=80, alpha=0.75)
plt.grid(True)
plt.gca().invert_yaxis()
plt.ylabel("y")
plt.xlabel("x")
plt.show()


X = points1.copy()

y_pred_vertical_lines = KMeans(n_clusters=7, random_state=1).fit_predict(X)

plt.figure(figsize=(8, 8), dpi=100)
plt.scatter(X[:, 0], X[:, 1], c=y_pred_vertical_lines, cmap='plasma_r', s=200, alpha=0.75)
plt.title("Clustering Vertical Lines (Bad)")
plt.show()


X = points1.copy()

# Let's Rotate the points by 60 degrees
theta = rotation_angles[0]*np.pi/180

# Define the rotation matrix
R = np.array([
    [np.cos(theta), np.sin(theta)],
    [-np.sin(theta), np.cos(theta)]])

# Rotate the points
X_rotated = X @ R.T

X_rotated[:,1] /= 100

y_pred_vertical_lines = KMeans(n_clusters=7, random_state=1).fit_predict(X_rotated)

plt.figure(figsize=(8, 8), dpi=100)
plt.scatter(X_rotated[:, 0], X_rotated[:, 1], c=y_pred_vertical_lines, cmap='plasma_r', s=200, alpha=0.75)
plt.title("Positive Lines (Scaled)")
plt.show()


X = points2.copy()

y_pred_positive_lines = KMeans(n_clusters=7, random_state=1).fit_predict(X)

plt.figure(figsize=(8, 8), dpi=100)
plt.scatter(X[:, 0], X[:, 1], c=y_pred_positive_lines, cmap='plasma_r', s=200, alpha=0.75)
plt.title("Positive Lines")
plt.show()


#Rotate the points by -60: Now they're like vertical lines
X = points2.copy()

# Let's Rotate the points by -60 degrees
theta = angle_limits[2]*np.pi/180

# Define the rotation matrix
R = np.array([
    [np.cos(theta), np.sin(theta)],
    [-np.sin(theta), np.cos(theta)]])
    
# Rotate the points
X_rotated = X @ R.T

y_pred_positive_lines = KMeans(n_clusters=7, random_state=1).fit_predict(X_rotated)

plt.figure(figsize=(8, 8), dpi=100)
plt.scatter(X_rotated[:, 0], X_rotated[:, 1], c=y_pred_positive_lines, cmap='plasma_r', s=200, alpha=0.75)
plt.title("Positive Lines (Rotated)")
plt.show()


#Divide the y component of the points by 10: Good
X = points2.copy()

# Let's Rotate the points by 60 degrees
theta = rotation_angles[1]*np.pi/180

# Define the rotation matrix
R = np.array([
    [np.cos(theta), np.sin(theta)],
    [-np.sin(theta), np.cos(theta)]])

# Rotate the points
X_rotated = X @ R.T

X_rotated[:,1] /= 100

y_pred_positive_lines = KMeans(n_clusters=7, random_state=1).fit_predict(X_rotated)

plt.figure(figsize=(8, 8), dpi=100)
plt.scatter(X_rotated[:, 0], X_rotated[:, 1], c=y_pred_positive_lines, cmap='plasma_r', s=200, alpha=0.75)
plt.title("Positive Lines (Rotated and Scaled)")
plt.show()


#Negative Lines
#Default clustering result: Bad
X = points3.copy()

y_pred_negative_lines = KMeans(n_clusters=7, random_state=1).fit_predict(X)

plt.figure(figsize=(8, 8), dpi=100)
plt.scatter(X[:, 0], X[:, 1], c=y_pred_negative_lines, cmap='plasma_r', s=200, alpha=0.75)
plt.title("Negative Lines (Bad)")
plt.show()


#Rotate the points by +60: Now they're like vertical lines
X = points3.copy()

# Let's Rotate the points by -60 degrees
theta = angle_limits[1]*np.pi/180

# Define the rotation matrix
R = np.array([
    [np.cos(theta), np.sin(theta)],
    [-np.sin(theta), np.cos(theta)]])
    
# Rotate the points
X_rotated = X @ R.T

y_pred_negative_lines = KMeans(n_clusters=7, random_state=1).fit_predict(X_rotated)

plt.figure(figsize=(8, 8), dpi=100)
plt.scatter(X_rotated[:, 0], X_rotated[:, 1], c=y_pred_negative_lines, cmap='plasma_r', s=200, alpha=0.75)
plt.title("Negative Lines (Rotated)")
plt.show()


# Divide the y component of the points by 10: Good
X = points3.copy()

# Let's Rotate the points by 60 degrees
theta = rotation_angles[2]*np.pi/180

# Define the rotation matrix
R = np.array([
    [np.cos(theta), np.sin(theta)],
    [-np.sin(theta), np.cos(theta)]])

# Rotate the points
X_rotated = X @ R.T

X_rotated[:,1] /= 100

y_pred_negative_lines = KMeans(n_clusters=7, random_state=1).fit_predict(X_rotated)

plt.figure(figsize=(8, 8), dpi=100)
plt.scatter(X_rotated[:, 0], X_rotated[:, 1], c=y_pred_negative_lines, cmap='plasma_r', s=200, alpha=0.75)
plt.title("Negative Lines (Rotated by +60 degrees and scaled y)")
plt.show()


#Line Fitting
#Vertical Lines
# Select only one cluster
cluster_id =4
X = points1.copy()[y_pred_vertical_lines==cluster_id]

# Create linear regression object
regr = LinearRegression()

# Train the model using the training sets
regr.fit(X[:,1].reshape(-1, 1), X[:,0])

# The coefficients
print("Coefficients: \n", regr.coef_[0])
print("Intercept: \n", regr.intercept_)

m = regr.coef_[0]
b = regr.intercept_

x = np.arange(0,img_3.shape[1])
y = m*x+b

plt.figure(figsize=(8, 8), dpi=100)
plt.scatter(X[:, 0], X[:, 1], c='g', cmap='plasma_r')
plt.plot(y, x)
plt.ylim([0,img_3.shape[0]])
plt.xlim([0,img_3.shape[1]])
plt.title("Fitted line y=mx+b")

plt.show()


fitted_ms_all = []
fitted_bs_all = []
scalers_all = []


class GHD_Scaler:
    def __init__(self):
        pass

    def fit(self, X):
        pass

    def transform(self, X):
        return X

    def inverse_transform(self, X):
        return X


fitted_ms = []
fitted_bs = []
scalers = []

# vertical lines
for cluster_id in range(7):
    X = points1.copy()[y_pred_vertical_lines==cluster_id]
    scaler = GHD_Scaler()
    scaler.fit(X)
    X = scaler.transform(X)
    scalers.append(scaler)

    regr = LinearRegression()
    regr.fit(X[:,1].reshape(-1, 1), X[:,0])

    m = regr.coef_[0]
    b = regr.intercept_

    fitted_ms.append(m)
    fitted_bs.append(b)

fitted_ms_all.append(fitted_ms)
fitted_bs_all.append(fitted_bs)
scalers_all.append(scalers)


plt.figure(figsize=(10, 10), dpi=100)
plt.imshow(img_3)
for cluster_id in range(7):
    # Get the points of this cluster
    X = points1.copy()[y_pred_vertical_lines==cluster_id]
    
    # Get the corresponding scaler
    scaler = scalers[cluster_id]

    # Calculate the points in the current line
    x = np.arange(0,img_3.shape[1])
    # Scale the x values so that they work with m and b
    x = scaler.transform(np.repeat(x[:,None], 2, axis=1))[:,0]
    y = fitted_ms[cluster_id]*x+fitted_bs[cluster_id]

    # Concatenate fitted line's x and y
    line_X = np.column_stack([y, x])
    
    # Inverse Scaler transform
    line_X = scaler.inverse_transform(line_X)

    plt.scatter(X[:, 0], X[:, 1], cmap='plasma_r', s=200, alpha=0.75)
    plt.plot(line_X[:, 0], line_X[:, 1], linewidth=3)

plt.ylim([0,img_3.shape[0]])
plt.xlim([0,img_3.shape[1]])
plt.gca().invert_yaxis()
plt.legend([1,2,3,4,5,6,7])
plt.title("Fitted lines")

plt.show()


#positive lines
fitted_ms = []
fitted_bs = []
scalers = []

# Positive lines
for cluster_id in range(7):
    X = points2.copy()[y_pred_positive_lines==cluster_id]
    scaler = GHD_Scaler()
    scaler.fit(X)
    X = scaler.transform(X)
    scalers.append(scaler)

    regr = LinearRegression()
    regr.fit(X[:,0].reshape(-1, 1), X[:,1])

    m = regr.coef_[0]
    b = regr.intercept_

    fitted_ms.append(m)
    fitted_bs.append(b)

fitted_ms_all.append(fitted_ms)
fitted_bs_all.append(fitted_bs)
scalers_all.append(scalers)


plt.figure(figsize=(10, 10), dpi=100)

plt.imshow(img_3)
for cluster_id in range(7):
    # Get the points of this cluster
    X = points2.copy()[y_pred_positive_lines==cluster_id]
    
    # Get the corresponding scaler
    scaler = scalers[cluster_id]

    # Calculate the points in the current line
    x = np.arange(0,img_3.shape[1])
    # Scale the x values so that they work with m and b
    x = scaler.transform(np.repeat(x[:,None], 2, axis=1))[:,0]
    y = fitted_ms[cluster_id]*x+fitted_bs[cluster_id]

    # Concatenate fitted line's x and y
    line_X = np.column_stack([x,y])
    
    # Inverse Scaler transform
    line_X = scaler.inverse_transform(line_X)

    plt.scatter(X[:, 0], X[:, 1], cmap='plasma_r', s=200, alpha=0.75)
    plt.plot(line_X[:, 0], line_X[:, 1], linewidth=3)

plt.ylim([0,img_3.shape[0]])
plt.xlim([0,img_3.shape[1]])
plt.gca().invert_yaxis()
plt.legend([1,2,3,4,5,6,7])
plt.title("Fitted lines")

plt.show()


#negative lines
fitted_ms = []
fitted_bs = []
scalers = []

# Negative lines
for cluster_id in range(7):
    X = points3.copy()[y_pred_negative_lines==cluster_id]
    scaler = GHD_Scaler()
    scaler.fit(X)
    X = scaler.transform(X)
    scalers.append(scaler)

    regr = LinearRegression()
    regr.fit(X[:,0].reshape(-1, 1), X[:,1])

    m = regr.coef_[0]
    b = regr.intercept_

    fitted_ms.append(m)
    fitted_bs.append(b)

fitted_ms_all.append(fitted_ms)
fitted_bs_all.append(fitted_bs)
scalers_all.append(scalers)


plt.figure(figsize=(10, 10), dpi=100)

plt.imshow(img_3)
for cluster_id in range(7):
    # Get the points of this cluster
    X = points3.copy()[y_pred_negative_lines==cluster_id]
    
    # Get the corresponding scaler
    scaler = scalers[cluster_id]

    # Calculate the points in the current line
    x = np.arange(0,img_3.shape[1])
    # Scale the x values so that they work with m and b
    x = scaler.transform(np.repeat(x[:,None], 2, axis=1))[:,0]
    y = fitted_ms[cluster_id]*x+fitted_bs[cluster_id]

    # Concatenate fitted line's x and y
    line_X = np.column_stack([x,y])
    
    # Inverse Scaler transform
    line_X = scaler.inverse_transform(line_X)

    plt.scatter(X[:, 0], X[:, 1], cmap='plasma_r', s=200, alpha=0.75)
    plt.plot(line_X[:, 0], line_X[:, 1], linewidth=3)

plt.ylim([0,img_3.shape[0]])
plt.xlim([0,img_3.shape[1]])
plt.gca().invert_yaxis()
plt.legend([1,2,3,4,5,6,7])
plt.title("Fitted lines")

plt.show()


#all together
plt.figure(figsize=(10, 10), dpi=100)

plt.imshow(img_3)

for i in range(3):
    for cluster_id in range(7):
        # Get the points of this cluster
        if i==0:
            X = points1.copy()[y_pred_vertical_lines==cluster_id]
        elif i==1:
            X = points2.copy()[y_pred_positive_lines==cluster_id]
        elif i==2:
            X = points3.copy()[y_pred_negative_lines==cluster_id]
        
        # Get the corresponding scaler
        scaler = scalers_all[i][cluster_id]

        # Calculate the points in the current line
        x = np.arange(0,img_3.shape[1])

        # Scale the x values so that they work with m and b
        x = scaler.transform(np.repeat(x[:,None], 2, axis=1))[:,0]
        y = fitted_ms_all[i][cluster_id]*x+fitted_bs_all[i][cluster_id]

        # Concatenate fitted line's x and y
        if i==0:
            line_X = np.column_stack([y,x])
        else:
            line_X = np.column_stack([x,y])
        
        # Inverse Scaler transform
        line_X = scaler.inverse_transform(line_X)

        # plt.scatter(X[:, 0], X[:, 1], cmap='plasma_r')
        plt.plot(line_X[:, 0], line_X[:, 1], c=colors_01[i], linewidth=3)

plt.ylim([0,img_3.shape[0]])
plt.xlim([0,img_3.shape[1]])
plt.gca().invert_yaxis()
# plt.legend([1,2,3,4,5,6,7])
plt.title("Fitted lines")

plt.show()


plt.figure(figsize=(10, 10), dpi=100)

plt.imshow(img_3_gray_rgb)

for i in range(3):
    for cluster_id in range(7):
        # Get the points of this cluster
        if i==0:
            X = points1.copy()[y_pred_vertical_lines==cluster_id]
        elif i==1:
            X = points2.copy()[y_pred_positive_lines==cluster_id]
        elif i==2:
            X = points3.copy()[y_pred_negative_lines==cluster_id]
        
        # Get the corresponding scaler
        scaler = scalers_all[i][cluster_id]

        # Calculate the points in the current line
        x = np.arange(0,img_3.shape[1])

        # Scale the x values so that they work with m and b
        x = scaler.transform(np.repeat(x[:,None], 2, axis=1))[:,0]
        y = fitted_ms_all[i][cluster_id]*x+fitted_bs_all[i][cluster_id]

        # Concatenate fitted line's x and y
        if i==0:
            line_X = np.column_stack([y,x])
        else:
            line_X = np.column_stack([x,y])
        
        # Inverse Scaler transform
        line_X = scaler.inverse_transform(line_X)

        # plt.scatter(X[:, 0], X[:, 1], cmap='plasma_r')
        plt.plot(line_X[:, 0], line_X[:, 1], c=colors_01[i], linewidth=3)

plt.ylim([0,img_3.shape[0]])
plt.xlim([0,img_3.shape[1]])
plt.gca().invert_yaxis()
# plt.legend([1,2,3,4,5,6,7])
plt.title("Fitted lines")

plt.show()


def plot_intersection_points_on_cube(fitted_ms_all, fitted_bs_all, scalers_all,  img_gray_rgb, direction_list=[1,2], debug=True, c="g"):
    points_on_the_face = []

    if debug:
        plt.figure(figsize=(8, 8), dpi=100)
        plt.imshow(img_gray_rgb)

    # Sort lines from left to right and bottom to top
    msi = fitted_ms_all[direction_list[0]]
    bsi = fitted_bs_all[direction_list[0]]
    scaler_i = scalers_all[direction_list[0]]

    # Sort lines by b
    if direction_list[0] == 0 and direction_list[1] == 2:
        sorted_indices_i = np.argsort(bsi)[::-1]
    elif direction_list[0] == 0 and direction_list[1] == 1:
        sorted_indices_i = np.argsort(bsi)
    else:
        sorted_indices_i = np.argsort(bsi)

    msi = np.array(msi)[sorted_indices_i]
    bsi = np.array(bsi)[sorted_indices_i]
    scaler_i = np.array(scaler_i)[sorted_indices_i]
    
    msj = fitted_ms_all[direction_list[1]]
    bsj = fitted_bs_all[direction_list[1]]
    scaler_j= scalers_all[direction_list[1]]

    # sort lines by b
    if direction_list[0] == 0 and direction_list[1] == 2:
        sorted_indices_j = np.argsort(bsj)[::-1]
    elif direction_list[0] == 0 and direction_list[1] == 1:
        sorted_indices_j = np.argsort(bsj)[::-1]
    else:
        sorted_indices_j = np.argsort(bsj)
    msj = np.array(msj)[sorted_indices_j]
    bsj = np.array(bsj)[sorted_indices_j]
    scaler_j = np.array(scaler_j)[sorted_indices_j]


    # first 4 lines of dir_a
    for i in range(4):
        # first 4 lines of dir_b
        for j in range(4):
            m1 = msi[i]
            b1 = bsi[i]
            m2 = msj[j]
            b2 = bsj[j]
            if direction_list[0] == 0:
                b1 = -b1/m1
                m1 = 1/m1
            
            x = (b2-b1)/(m1-m2)
            y = m1*x+b1

            points_on_the_face.append([x,y])

            if debug:
                plt.scatter([x], [y], c=c)

    if debug:
        plt.ylim([0,img_gray_rgb.shape[0]])
        plt.xlim([0,img_gray_rgb.shape[1]])
        plt.gca().invert_yaxis()
        plt.title("Fitted lines")

        plt.show()

    return points_on_the_face


def display_reconstructed_faces(reconstructed_faces):
    faces_names = ["Left", "Right", "Top"]
    for f in range(3):
        disp(reconstructed_faces[f],  s=3, title=faces_names[f])


img_gray = img_3_gray
img = img_3
img_gray_rgb = np.repeat(img_gray[:, :, None], 3, 2)


points_on_left_face = plot_intersection_points_on_cube(
    fitted_ms_all, fitted_bs_all, scalers_all,  img_gray_rgb,
    direction_list=[0, 1], debug=True)


points_on_right_face = plot_intersection_points_on_cube(
    fitted_ms_all, fitted_bs_all, scalers_all,  img_gray_rgb,
    direction_list=[0, 2], debug=True, c='r')


points_on_top_face = plot_intersection_points_on_cube(
    fitted_ms_all, fitted_bs_all, scalers_all,  img_gray_rgb,
    direction_list=[1, 2], debug=True, c='b')



list_of_colors = [[255, 30, 30], [0, 255, 0], [0, 0, 255],
                  [255, 110, 0], [255, 200, 0], [255, 255, 255]]
list_of_colors_names = ["red", "green", "blue", "orange", "yellow", "white"]


def closest(color):
    list_of_colors_val = np.array(list_of_colors)
    color = np.array(color)
    distances = np.sqrt(np.sum((list_of_colors_val-color)**2, axis=1))
    index_of_smallest = np.where(distances == np.amin(distances))
    smallest_distance = list_of_colors_val[index_of_smallest][0]
    return smallest_distance,list_of_colors_names[index_of_smallest[0][0]]


# 11. Find face centers
points_left = np.array(points_on_left_face)
points_right = np.array(points_on_right_face)
points_top = np.array(points_on_top_face)

face_indices = [
    [2, 3, 7, 6], [6, 7, 11, 10], [10, 11, 15, 14],
    [1, 2, 6, 5], [5, 6, 10, 9], [9, 10, 14, 13],
    [0, 1, 5, 4], [4, 5, 9, 8], [8, 9, 13, 12]
]

face_centers = [[], [], []]

plt.figure(figsize=(8, 8), dpi=100)
plt.imshow(img_gray_rgb)

for face in face_indices:
    face_center = (points_left[face[0]] + points_left[face[1]] +
                points_left[face[2]] + points_left[face[3]]) / 4
    face_centers[0].append(face_center)

    plt.scatter([face_center[0]], [face_center[1]], c='r', s=86)

for face in face_indices:
    face_center = (points_right[face[0]] + points_right[face[1]] +
                points_right[face[2]] + points_right[face[3]]) / 4
    face_centers[1].append(face_center)

    plt.scatter([face_center[0]], [face_center[1]], c='g', s=86)

for face in face_indices:
    face_center = (points_top[face[0]] + points_top[face[1]] +
                points_top[face[2]] + points_top[face[3]]) / 4
    face_centers[2].append(face_center)

    plt.scatter([face_center[0]], [face_center[1]], c='b', s=86)

# 12. Extract face colors
reconstructed_faces = []
reconstructed_faces_closest = []

faces_names = ["Left", "Right", "Top"]

# disp(img)
for f in range(3):
    reconstructed_face = np.zeros((3, 3, 3), dtype=np.uint8)
    reconstructed_face_closest=np.zeros((3, 3, 3), dtype=np.uint8)
    for i in range(9):
        x, y = face_centers[f][i]
        x, y = int(x), int(y)
        w = 40
        mean_color = img[y-w//2:y+w//2, x-w//2:x +
                        w//2].mean(axis=(0, 1)).astype(np.uint8)
        reconstructed_face[i//3, i % 3, :] = mean_color
        reconstructed_face_closest[i//3, i % 3, :],detected_color=closest(mean_color)
        print(mean_color,detected_color)


    reconstructed_faces.append(reconstructed_face)
    reconstructed_faces_closest.append(reconstructed_face_closest)
    
# Fix face orientations
# Right face
reconstructed_faces[1] = np.flip(reconstructed_faces[1], axis=1)
# Top face
reconstructed_faces[2] = np.flip(reconstructed_faces[2], axis=1)
reconstructed_faces[2] = np.flip(reconstructed_faces[2], axis=0)

display_reconstructed_faces(reconstructed_faces)


print("******************")
reconstructed_faces_closest[1] = np.flip(reconstructed_faces_closest[1], axis=1)
# Top face
reconstructed_faces_closest[2] = np.flip(reconstructed_faces_closest[2], axis=1)
reconstructed_faces_closest[2] = np.flip(reconstructed_faces_closest[2], axis=0)

display_reconstructed_faces(reconstructed_faces_closest)


