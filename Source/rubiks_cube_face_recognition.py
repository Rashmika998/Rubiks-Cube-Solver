from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from ipywidgets import interact, interactive, fixed, interact_manual
import time
import numpy as np
import cv2
from tqdm import trange
import matplotlib.pyplot as plt
import seaborn as sns
import glob
from math import sqrt

sns.set()
sns.set_style('dark')
RECOGNIZED_IMAGES = "recognized images/"
INPUT_IMAGES = "Input/"


def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


list_of_colors = [[255, 30, 30], [0, 255, 0], [0, 0, 255],
                  [255, 110, 0], [255, 200, 0], [255, 255, 255]]
list_of_colors_names = ["red", "green", "blue", "orange", "yellow", "white"]


def closest(color,hsv):
    color = "red"
    if int(hsv[0]) < 5:
        if int(hsv[1] < 20):
            color = "white"
        else:
            color = "red"
    elif int(hsv[0]) < 22:
        color = "orange"
    elif int(hsv[0]) < 33:
        color = "yellow"
        # else:
        #     color = "red"
    elif int(hsv[0]) < 78:
        color = "green"
    elif int(hsv[0]) < 115:
        if int(hsv[1] > 200):
            color = "blue"
        elif int(hsv[1] < 20):
            color = "white"
    else:
        if int(hsv[1] < 20):
            color = "white"
        else:
            color = "red"
    if color == "red":
        smallest_distance = [255, 30, 30]
    elif color == "white":
        smallest_distance = [255, 255, 255]
    elif color == "orange":
        smallest_distance = [255, 110, 0]
    elif color == "yellow":
        smallest_distance = [255, 255, 0]
    elif color == "green":
        smallest_distance = [0, 255, 0]
    elif color == "blue":
        smallest_distance = [0, 0, 255]
    # list_of_colors_val = np.array(list_of_colors)
    # color = np.array(color)
    # distances = np.sqrt(np.sum((list_of_colors_val-color)**2, axis=1))
    # index_of_smallest = np.where(distances == np.amin(distances))
    # smallest_distance = list_of_colors_val[index_of_smallest][0]
    return smallest_distance, color


class GHD_Scaler:
    def __init__(self):
        pass

    def fit(self, X):
        pass

    def transform(self, X):
        return X

    def inverse_transform(self, X):
        return X


def cluster_points(points, rotate_degree=0, squash_factor=100, n_clusters=7, debug=False):
    X = points.copy()

    # if rotate_degree != 0:
        # Let's Rotate the points by 'rotate_degree' degrees
    theta = rotate_degree*np.pi/180

        # Define the rotation matrix
    R = np.array([
            [np.cos(theta), np.sin(theta)],
            [-np.sin(theta), np.cos(theta)]])

        # Rotate the points
    X = X @ R.T

    X[:, 1] /= squash_factor

    cluster_ids = KMeans(n_clusters=n_clusters, random_state=1).fit_predict(X)

    # plt.figure(figsize=(8, 8), dpi=100)
    # plt.scatter(X[:, 0], X[:, 1], c=cluster_ids, cmap='plasma_r', s=200, alpha=0.75)
    # plt.title(" Lines (Rotated and Scaled)")
    # plt.show()

    # if debug:
    #     plt.figure(figsize=(6, 6))
    #     plt.scatter(X[:, 0], X[:, 1], c=cluster_ids, cmap='plasma_r')
    #     plt.title(
    #         f"Positive Lines (Rotated by {rotate_degree} degrees and scaled y by {squash_factor})")
    #     plt.show()

    return cluster_ids


def line_to_points(lines):
    points = np.zeros((len(lines)*2, 2))
    for i in range(len(lines)):
        # point A on a line
        points[2*i][0] = (lines[i][0])
        points[2*i][1] = (lines[i][1])
        # point B on a line
        points[2*i+1][0] = (lines[i][2])
        points[2*i+1][1] = (lines[i][3])
    return points


def plot_lines_on_cube(points_1, points_2, points_3, fitted_ms_all, fitted_bs_all, scalers_all,  img_gray_rgb, y_pred_1, y_pred_2, y_pred_3, direction_list=[0, 1, 2], n_clusters=7):
    # plt.figure(figsize=(6, 6))
    # plt.imshow(img_gray_rgb)

    for i in direction_list:
        for cluster_id in range(n_clusters):
            # Get the points of this cluster
            if i == 0:
                X = points_1.copy()[y_pred_1 == cluster_id]
            elif i == 1:
                X = points_2.copy()[y_pred_2 == cluster_id]
            elif i == 2:
                X = points_3.copy()[y_pred_3 == cluster_id]

            # Get the corresponding scaler
            scaler = scalers_all[i][cluster_id]

            # Calculate the points in the current line
            x = np.arange(0, img_gray_rgb.shape[1])

            # Scale the x values so that they work with m and b
            x = scaler.transform(np.repeat(x[:, None], 2, axis=1))[:, 0]
            y = fitted_ms_all[i][cluster_id]*x+fitted_bs_all[i][cluster_id]

            # Concatenate fitted line's x and y
            if i == 0:
                # if vertical: x=my+b
                line_X = np.column_stack([y, x])
            else:
                # else: y=mx+b
                line_X = np.column_stack([x, y])

            # Inverse Scaler transform
            line_X = scaler.inverse_transform(line_X)

            # plt.plot(line_X[:, 0], line_X[:, 1], c=colors_01[i], linewidth=1.5)

    # plt.ylim([0, img_gray_rgb.shape[0]])
    # plt.xlim([0, img_gray_rgb.shape[1]])
    # plt.gca().invert_yaxis()
    # plt.title("Fitted lines")

    # plt.show()


def plot_intersection_points_on_cube(fitted_ms_all, fitted_bs_all, scalers_all,  img_gray_rgb, direction_list=[1, 2], debug=True, c="g"):
    points_on_the_face = []

    # if debug:
    #     plt.figure(figsize=(6, 6))
    #     plt.imshow(img_gray_rgb)

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
    scaler_j = scalers_all[direction_list[1]]

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

            points_on_the_face.append([x, y])

            # if debug:
            # plt.scatter([x], [y], c=c)

    # if debug:
    #     plt.ylim([0, img_gray_rgb.shape[0]])
    #     plt.xlim([0, img_gray_rgb.shape[1]])
    #     plt.gca().invert_yaxis()
    #     plt.title("Fitted lines")

    #     plt.show()

    return points_on_the_face


def fit_lines(img,points, y_pred, n_clusters=7, is_vertical=False):
    fitted_ms = []
    fitted_bs = []
    scalers = []

    for cluster_id in range(n_clusters):
        X = points.copy()[y_pred == cluster_id]

        # Scale features
        scaler = GHD_Scaler()
        scaler.fit(X)
        X = scaler.transform(X)
        scalers.append(scaler)

        # Fit a line to the points using linear regression
        regr = LinearRegression()

        # If 'is_vertical': fit x=my+b instead of y=mx+b
        if is_vertical:
            regr.fit(X[:, 1].reshape(-1, 1), X[:, 0])
        else:
            regr.fit(X[:, 0].reshape(-1, 1), X[:, 1])

        m = regr.coef_[0]
        b = regr.intercept_

        fitted_ms.append(m)
        fitted_bs.append(b)

    # plt.figure(figsize=(10, 10), dpi=100)
    # plt.imshow(img)
    for cluster_id in range(7):
            # Get the points of this cluster
        X = points.copy()[y_pred==cluster_id]
            
            # Get the corresponding scaler
        scaler = scalers[cluster_id]

            # Calculate the points in the current line
        x = np.arange(0,img.shape[1])
            # Scale the x values so that they work with m and b
        x = scaler.transform(np.repeat(x[:,None], 2, axis=1))[:,0]
        y = fitted_ms[cluster_id]*x+fitted_bs[cluster_id]

            # Concatenate fitted line's x and y

        if is_vertical:
            line_X = np.column_stack([y, x])
        else:
            line_X = np.column_stack([x, y])
            
            # Inverse Scaler transform
        line_X = scaler.inverse_transform(line_X)

    #     plt.scatter(X[:, 0], X[:, 1], cmap='plasma_r', s=200, alpha=0.75)
    #     plt.plot(line_X[:, 0], line_X[:, 1], linewidth=3)

    # plt.ylim([0,img.shape[0]])
    # plt.xlim([0,img.shape[1]])
    # plt.gca().invert_yaxis()
    # plt.legend([1,2,3,4,5,6,7])
    # plt.title("Fitted lines")

    # plt.show()        

    return fitted_ms, fitted_bs, scalers
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

def extract_faces(img, img_gray, kernel_size=5, canny_low=0, canny_high=75, min_line_length=40, max_line_gap=20, center_sampling_width=10, colors=None, colors_01=None, n_clusters=7, debug=False, debug_time=True,turn=1,imgName=""):
    """Takes an image of a rubiks cube, finds edges, fits lines to edges and extracts the faces

    Args:
        img (RGB image): rubiks cube image
    """
    detected_cols_list_sides = []  # list to save the colors names of the cube sides
    # list to save the colors names of the cube top and bottom
    detected_cols_list_top_bottom = []
    start_time = time.perf_counter_ns()

    temp_img = img
    # 1. Reduce the noises in the original image
    img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    img_gray_denoise = cv2.fastNlMeansDenoising(img_gray,None,20,7,21)

    gray = cv2.morphologyEx(img_gray_denoise, cv2.MORPH_OPEN, kernel)

    # 2. Smooth it using Gaussian Blur and reduce the noise again
    blur_gray = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)

    divide = shadow_remove(blur_gray)
    divide = cv2.fastNlMeansDenoising(divide,None,10,7,21)

    # blur_gray = cv2.fastNlMeansDenoising(blur_gray, None, 10, 7, 21)

    # 3. Canny
    edges = cv2.Canny(divide, canny_low, canny_high,apertureSize = 5, 
                 L2gradient = True)

    # disp(edges)

    # plt.imshow(edges)
    # plt.show()

    # 4. HoughLinesP
    # Distance resolution in pixels of the Hough grid
    rho = 1
    # Angular resolution in radians of the Hough grid
    theta = np.pi / 180
    # Other Hough params
    threshold = 15
    min_line_length = min_line_length
    max_line_gap = max_line_gap
    line_image = np.copy(img) * 0

    # Run Hough on edge detected image
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

    # 5. Calculate Angles
    if lines is not None:
        # print(len(lines),"lines detected")

        angles = []
        for line in lines:
            for x1, y1, x2, y2 in line:
                # Calculate the angle
                angle = cv2.fastAtan2(float(y2-y1), float(x2-x1))

                # 210 == 180+30 ==> 30
                if angle >= 180:
                    angle -= 180
                angles.append(angle)
                cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)

        img_3_gray_rgb = np.repeat(img_gray[:,:,None], 3, 2)
        lines_edges = cv2.addWeighted(img_3_gray_rgb, 0.8, line_image, 1, 0)

        # disp(lines_edges)
        angles = np.array(angles)
        

        # Cluster the angles to find the breaking points
        angles_clustering = KMeans(n_clusters=3, n_init=2)
        angles_clustering.fit(angles.reshape(-1, 1))
        sorted_centers = sorted(angles_clustering.cluster_centers_)

        # Used to split lines into vertical/+ve/-ve categories
        angle_limits = [0, 60, 120, 180]
        angle_limits[1] = ((sorted_centers[0] + sorted_centers[1]) // 2)[0]
        angle_limits[2] = ((sorted_centers[1] + sorted_centers[2]) // 2)[0]

        # used in step 7 to orient the clusters
        rotation_angles = [0, 60, 120]
        rotation_angles[0] = -(90 - sorted_centers[1])[0]  # vertical
        rotation_angles[1] = -(90 - sorted_centers[0])[0]  # positive
        rotation_angles[2] = -(90 - sorted_centers[2])[0]  # negative

        # 5.5 Calculate Angles
        angles = []
        vertical_lines = []
        negative_lines = []
        positive_lines = []
        for line in lines:
            for x1, y1, x2, y2 in line:
                # Calculate the angle
                angle = cv2.fastAtan2(float(y2-y1), float(x2-x1))
                angles.append(angle)

                # 210 == 180+30 == 30
                if angle >= 180:
                    angle -= 180
                angles.append(angle)

                # Find the type of the line
                if angle_limits[0] <= angle <= angle_limits[1]:
                    cluster_id = 0
                    positive_lines.append([x1, y1, x2, y2])
                if angle_limits[1] <= angle <= angle_limits[2]:
                    cluster_id = 1
                    vertical_lines.append([x1, y1, x2, y2])
                if angle_limits[2] <= angle <= angle_limits[3]:
                    cluster_id = 2
                    negative_lines.append([x1, y1, x2, y2])

                cv2.line(line_image, (x1, y1), (x2, y2), colors[cluster_id], 5)

        img_3_gray_rgb = np.repeat(img_gray[:,:,None], 3, 2)
        lines_edges = cv2.addWeighted(img_3_gray_rgb, 1, line_image, 1, 0)

        # disp(lines_edges)
        # 6. Lines to points
        points_1 = line_to_points(vertical_lines)
        points_2 = line_to_points(positive_lines)
        points_3 = line_to_points(negative_lines)


        # plt.figure(figsize=(8,8), dpi=100)
        # plt.title("All of the lines")
        # plt.scatter(points_1[:,0], points_1[:,1],c='g', s=80, alpha=0.75)
        # plt.scatter(points_2[:,0], points_2[:,1],c='r', s=80, alpha=0.75)
        # plt.scatter(points_3[:,0], points_3[:,1],c='b', s=80, alpha=0.75)
        # plt.grid(True)
        # plt.gca().invert_yaxis()
        # plt.ylabel("y")
        # plt.xlabel("x")
        # plt.show()
        # 7. Cluster points
        y_pred_1 = cluster_points(
            points_1, rotate_degree=rotation_angles[0], squash_factor=100, debug=debug)
        y_pred_2 = cluster_points(
            points_2, rotate_degree=rotation_angles[1], squash_factor=100, debug=debug)

        y_pred_3 = cluster_points(
            points_3, rotate_degree=rotation_angles[2], squash_factor=100, debug=debug)

        # 8. Line Fitting (y=mx+b) or (x=my+b)
        fitted_ms_1, fitted_bs_1, scalers_1 = fit_lines(img,
            points_1, y_pred_1, n_clusters=n_clusters, is_vertical=True)
        fitted_ms_2, fitted_bs_2, scalers_2 = fit_lines(img,
            points_2, y_pred_2, n_clusters=n_clusters, is_vertical=False)
        fitted_ms_3, fitted_bs_3, scalers_3 = fit_lines(img,
            points_3, y_pred_3, n_clusters=n_clusters, is_vertical=False)


        

        fitted_ms_all = [fitted_ms_1, fitted_ms_2, fitted_ms_3]
        fitted_bs_all = [fitted_bs_1, fitted_bs_2, fitted_bs_3]
        scalers_all = [scalers_1, scalers_2, scalers_3]

        #all together
        # plt.figure(figsize=(10, 10), dpi=100)

        # plt.imshow(img)

        for i in range(3):
            for cluster_id in range(7):
                # Get the points of this cluster
                if i==0:
                    X = points_1.copy()[y_pred_1==cluster_id]
                elif i==1:
                    X = points_2.copy()[y_pred_2==cluster_id]
                elif i==2:
                    X = points_3.copy()[y_pred_3==cluster_id]
                
                # Get the corresponding scaler
                scaler = scalers_all[i][cluster_id]

                # Calculate the points in the current line
                x = np.arange(0,img.shape[1])

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
        #         plt.plot(line_X[:, 0], line_X[:, 1], c=colors_01[i], linewidth=3)

        # plt.ylim([0,img.shape[0]])
        # plt.xlim([0,img.shape[1]])
        # plt.gca().invert_yaxis()
        # # plt.legend([1,2,3,4,5,6,7])
        # plt.title("Fitted lines")

        # plt.show()


        # 10. Find intersection points

        img_gray_rgb = np.repeat(img_gray[:, :, None], 3, 2)
        points_on_left_face = plot_intersection_points_on_cube(
            fitted_ms_all, fitted_bs_all, scalers_all,  img_gray_rgb,
            direction_list=[0, 1], debug=debug)

        points_on_right_face = plot_intersection_points_on_cube(
            fitted_ms_all, fitted_bs_all, scalers_all,  img_gray_rgb,
            direction_list=[0, 2], debug=debug)

        points_on_top_face = plot_intersection_points_on_cube(
            fitted_ms_all, fitted_bs_all, scalers_all,  img_gray_rgb,
            direction_list=[1, 2], debug=debug)

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
        # plt.figure(figsize=(8, 8), dpi=100)

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

        plt.show()
        # print("*****************"+imgName)
        # cv2.imwrite(RECOGNIZED_IMAGES+imgName.split("/")[1].split(".")[0]+".png", img_gray_rgb)
        # 12. Extract face colors
        reconstructed_faces = []
        faces_names = ["Left", "Right", "Top"]

        for f in range(3):
            reconstructed_face = np.zeros((3, 3, 3), dtype=np.uint8)
            for i in range(9):
                x, y = face_centers[f][i]
                x, y = int(x), int(y)
                w = center_sampling_width
                mean_color = img[y-w//2:y+w//2, x-w//2:x +
                                 w//2].mean(axis=(0, 1)).astype(np.uint8)
                hsv_mean=cv2.cvtColor(np.uint8([[mean_color]]),cv2.COLOR_RGB2HSV).mean(axis=(0,1)).astype(np.uint8)
                reconstructed_face[i//3, i %
                                   3, :], detected_color = closest(mean_color,hsv_mean)
                # print(mean_color,closest(mean_color))
                if f == 2:
                    detected_cols_list_top_bottom.append(
                        detected_color)  # save the side colors
                else:
                    detected_cols_list_sides.append(
                        detected_color)  # save the top colors

            # swap the list elements according to the format used in algorithm part
            if turn==1:
                if f == 0:
                    detected_cols_list_sides[0], detected_cols_list_sides[1], detected_cols_list_sides[2], detected_cols_list_sides[
                    3], detected_cols_list_sides[4], detected_cols_list_sides[5], detected_cols_list_sides[6], detected_cols_list_sides[7], detected_cols_list_sides[8] = detected_cols_list_sides[6], detected_cols_list_sides[3], detected_cols_list_sides[0], detected_cols_list_sides[
                    7], detected_cols_list_sides[4], detected_cols_list_sides[1], detected_cols_list_sides[8], detected_cols_list_sides[5], detected_cols_list_sides[2]
                elif f == 1:
                    detected_cols_list_sides[9], detected_cols_list_sides[10], detected_cols_list_sides[11], detected_cols_list_sides[
                    12], detected_cols_list_sides[13], detected_cols_list_sides[14], detected_cols_list_sides[15], detected_cols_list_sides[16], detected_cols_list_sides[17] = detected_cols_list_sides[17], detected_cols_list_sides[14], detected_cols_list_sides[11], detected_cols_list_sides[
                    16], detected_cols_list_sides[13], detected_cols_list_sides[10], detected_cols_list_sides[15], detected_cols_list_sides[12], detected_cols_list_sides[9]
                else:
                    detected_cols_list_top_bottom[0], detected_cols_list_top_bottom[1], detected_cols_list_top_bottom[2], detected_cols_list_top_bottom[
                    3], detected_cols_list_top_bottom[4], detected_cols_list_top_bottom[5], detected_cols_list_top_bottom[6], detected_cols_list_top_bottom[7], detected_cols_list_top_bottom[8] = detected_cols_list_top_bottom[8], detected_cols_list_top_bottom[7], detected_cols_list_top_bottom[6], detected_cols_list_top_bottom[
                    5], detected_cols_list_top_bottom[4], detected_cols_list_top_bottom[3], detected_cols_list_top_bottom[2], detected_cols_list_top_bottom[1], detected_cols_list_top_bottom[0]
            elif turn==2:
                if f == 0:
                    detected_cols_list_sides[0], detected_cols_list_sides[1], detected_cols_list_sides[2], detected_cols_list_sides[
                    3], detected_cols_list_sides[4], detected_cols_list_sides[5], detected_cols_list_sides[6], detected_cols_list_sides[7], detected_cols_list_sides[8] = detected_cols_list_sides[2], detected_cols_list_sides[5], detected_cols_list_sides[8], detected_cols_list_sides[
                    1], detected_cols_list_sides[4], detected_cols_list_sides[7], detected_cols_list_sides[0], detected_cols_list_sides[3], detected_cols_list_sides[6]
                elif f == 1:
                    detected_cols_list_sides[9], detected_cols_list_sides[10], detected_cols_list_sides[11], detected_cols_list_sides[
                    12], detected_cols_list_sides[13], detected_cols_list_sides[14], detected_cols_list_sides[15], detected_cols_list_sides[16], detected_cols_list_sides[17] = detected_cols_list_sides[9], detected_cols_list_sides[12], detected_cols_list_sides[15], detected_cols_list_sides[
                    10], detected_cols_list_sides[13], detected_cols_list_sides[16], detected_cols_list_sides[11], detected_cols_list_sides[14], detected_cols_list_sides[17]
                else:
                    detected_cols_list_top_bottom[0], detected_cols_list_top_bottom[1], detected_cols_list_top_bottom[2], detected_cols_list_top_bottom[
                    3], detected_cols_list_top_bottom[4], detected_cols_list_top_bottom[5], detected_cols_list_top_bottom[6], detected_cols_list_top_bottom[7], detected_cols_list_top_bottom[8] = detected_cols_list_top_bottom[2], detected_cols_list_top_bottom[5], detected_cols_list_top_bottom[8], detected_cols_list_top_bottom[
                    1], detected_cols_list_top_bottom[4], detected_cols_list_top_bottom[7], detected_cols_list_top_bottom[0], detected_cols_list_top_bottom[3], detected_cols_list_top_bottom[6]

                    # swap two sides
                    detected_cols_list_sides[0], detected_cols_list_sides[1], detected_cols_list_sides[2], detected_cols_list_sides[
                    3], detected_cols_list_sides[4], detected_cols_list_sides[5], detected_cols_list_sides[6], detected_cols_list_sides[7], detected_cols_list_sides[8],detected_cols_list_sides[9], detected_cols_list_sides[10], detected_cols_list_sides[11], detected_cols_list_sides[
                    12], detected_cols_list_sides[13], detected_cols_list_sides[14], detected_cols_list_sides[15], detected_cols_list_sides[16], detected_cols_list_sides[17] = detected_cols_list_sides[9], detected_cols_list_sides[10], detected_cols_list_sides[11], detected_cols_list_sides[
                    12], detected_cols_list_sides[13], detected_cols_list_sides[14], detected_cols_list_sides[15], detected_cols_list_sides[16], detected_cols_list_sides[17],detected_cols_list_sides[0], detected_cols_list_sides[1], detected_cols_list_sides[2], detected_cols_list_sides[
                    3], detected_cols_list_sides[4], detected_cols_list_sides[5], detected_cols_list_sides[6], detected_cols_list_sides[7], detected_cols_list_sides[8]
            reconstructed_faces.append(reconstructed_face)

        # Fix face orientations
        # Right face
        reconstructed_faces[1] = np.flip(reconstructed_faces[1], axis=1)
        # Top face
        reconstructed_faces[2] = np.flip(reconstructed_faces[2], axis=1)
        reconstructed_faces[2] = np.flip(reconstructed_faces[2], axis=0)
        return reconstructed_faces, detected_cols_list_sides, detected_cols_list_top_bottom


# Main
colors = [
    (245, 0, 87),  # rgb(245, 0, 87)
    (0, 230, 118),  # rgb(0, 230, 118)
    (25, 118, 210),  # rgb(25, 118, 210)
    (245, 124, 0),  # rgb(245, 124, 0)
    (124, 77, 255)  # rgb(124, 77, 255)
]

colors_01 = [
    (245/255, 0/255, 87/255),  # rgb(245, 0, 87)
    (0/255, 230/255, 118/255),  # rgb(0, 230, 118)
    (25/255, 118/255, 210/255),  # rgb(25, 118, 210)
    (245/255, 124/255, 0/255),  # rgb(245, 124, 0)
    (124/255, 77/255, 255/255)  # rgb(124, 77, 255)
]


def detect_colors():
    # reconstructed_faces, detected_cols_list_sides, detected_cols_list_top_bottom = extract_faces(
    #     img,
    #     image_grey,
    #     kernel_size=7,
    #     canny_low=0,
    #     canny_high=75,
    #     min_line_length=40,
    #     max_line_gap=20,
    #     center_sampling_width=40,
    #     colors=colors,
    #     colors_01=colors_01,
    #     n_clusters=7,
    #     debug=True,
    #     debug_time=True)
    i = 1
    detect_colors_list = []  # list to store the side colors
    # temporary list to store the top/bottom colors
    detect_colors_list_top_bottom = []
    for img in glob.glob(INPUT_IMAGES+'*'):
        input_image = rgb(cv2.imread(img))
        input_image_grey = cv2.imread(img, 0)
        detected_image_sides, temp_detected_cols_list_sides, temp_detected_cols_list_top_bottom = extract_faces(
            input_image,
            input_image_grey,
            kernel_size=7,
            canny_low=0,
            canny_high=75,
            min_line_length=40,
            max_line_gap=20,
            center_sampling_width=40,
            colors=colors,
            colors_01=colors_01,
            n_clusters=7,
            debug=True,
            debug_time=True,
            turn=i,
            imgName=img)
        for j in range(len(detected_image_sides)):
            detected_image_sides[j] = cv2.cvtColor(
                detected_image_sides[j], cv2.COLOR_BGR2RGB)  # convert the bgr image to rgb
            detected_image_sides[j] = cv2.resize(
                detected_image_sides[j], (500, 500), interpolation=cv2.INTER_AREA)  # resize it to display more clearly

            # save the 6 sides of the images in the given directory
            if i == 1:
                cv2.imwrite(RECOGNIZED_IMAGES+str(j+1) +
                            ".png", detected_image_sides[j])
            elif i == 2:
                cv2.imwrite(RECOGNIZED_IMAGES+str(j+4) +
                            ".png", detected_image_sides[j])
        detected_image_sides.clear()
        detect_colors_list.extend(temp_detected_cols_list_sides)
        detect_colors_list_top_bottom.extend(
            temp_detected_cols_list_top_bottom)
        temp_detected_cols_list_sides.clear()
        temp_detected_cols_list_top_bottom.clear()
        i = i+1

    # The list index format in the algorithm code is 4 sides, top, bottom. Hence extend the top bottom color values to the 4 sides list color values
    detect_colors_list.extend(detect_colors_list_top_bottom)
    return detect_colors_list

if __name__ == '__main__':
    detect_colors()
