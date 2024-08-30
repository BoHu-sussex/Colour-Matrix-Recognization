import cv2
import numpy as np
import math
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans


def find_dense_black_clusters(image):
    """
    Find the clusters of dense black pixels.
    :param image: Grayscale map with white pixels as the background and black pixels as the foreground.
    :return: A list of tuples. Each tuple is a coordinate representing the center of the cluster.
    """

    # Get the coordinate of black pixels.
    black_pixels = np.argwhere(image == 0)

    # Convert the coordinates to a format suitable for DBSCAN. The coordinates of OpenCV images are (y, x),
    # while the usual data processing is (x, y).
    X = black_pixels[:, [1, 0]]

    # Apply DBSCAN clustering.
    db = DBSCAN(eps=5, min_samples=3).fit(X)

    # Get clustering results.
    labels = db.labels_
    unique_labels = set(labels)

    clusters = []
    for k in unique_labels:
        # -1 indicates the noise point.
        if k == -1:
            continue
        class_member_mask = (labels == k)
        xy = X[class_member_mask & (labels != -1)]
        # Get the center of mass of the cluster as the representative coordinates.
        c = xy.mean(axis=0)
        clusters.append(tuple(map(int, c)))

    return clusters


def calculate_center(points):
    """
    Calculate all the centers of points.
    :param points: A list of points.
    :return: The center of these list of points.
    """
    x_sum, y_sum = sum(x for x, y in points), sum(y for x, y in points)
    center_x, center_y = x_sum / len(points), y_sum / len(points)
    return center_x, center_y


def polar_angle(point, center):
    """
    Calculate the polar Angle of the point with respect to the center.
    :param point: The coordinate of a point.
    :param center: The center of a list of points.
    :return: The polar Angle of the point with respect to the center.
    """
    dx, dy = point[0] - center[0], point[1] - center[1]
    angle = math.atan2(dy, dx)
    # Convert the Angle to a range of 0 to 2π, ensuring that the clockwise ordering is correct.
    if angle < 0:
        angle += 2 * math.pi
    return angle


def sort_points_clockwise(points):
    """
    Sort the coordinate points in clockwise order.
    :param points: A list of points.
    :return: Points listed in clockwise order.
    """
    center = calculate_center(points)
    sorted_points = sorted(points, key=lambda p: polar_angle(p, center), reverse=True)
    return sorted_points


def getCircle(image_path):
    """
    Get the coordinates to locate the black circle around the image.
    :param image_path: The path of image.
    :return: The coordinates of the four black circles around the image.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Use a threshold to process the image.
    thresh = 20
    _, threshold = cv2.threshold(image, thresh, 255, cv2.THRESH_BINARY)

    # Find the clusters of black pixels.
    clusters = find_dense_black_clusters(threshold)

    # Divide clusters into four categories via k-means algorithm.
    coordinates_np = np.array(clusters)
    kmeans = KMeans(n_clusters=4, random_state=0)
    kmeans.fit(coordinates_np)

    # Gets the center coordinates for each category.
    cluster_centers = kmeans.cluster_centers_
    corners = []
    for i, center in enumerate(cluster_centers):
        x, y = center
        x, y = int(x), int(y)
        corners.append((x, y))

    # Order the four coordinates clockwise.
    sorted_corners = sort_points_clockwise(corners)
    return sorted_corners


def transform(image_path, trans, org=None):
    """
    Restore perspective transformed images to their original shape.
    :param image_path: The path of image.
    :param trans: The coordinates of the four anchor points of the image after perspective transformation.
    :param org: The coordinates of the four anchor points of the original image.
    :return: The image after recovery.
    """
    # Define the four corner coordinates of the transformed rectangle.
    if org is None:
        org = [(0, 0), (0, 480), (480, 480), (480, 0)]

    original_corners = np.float32(org)
    transformed_corners = np.float32(trans)

    # Calculate perspective transformation matrix.
    m = cv2.getPerspectiveTransform(transformed_corners, original_corners)

    # Read the transformed image.
    image = cv2.imread(image_path)

    # Apply the reverse perspective transform.
    restored_image = cv2.warpPerspective(image, m, (image.shape[1], image.shape[0]))

    # Crop image.
    restored_image = restored_image[50:430, 50:430]

    return restored_image


def saturate_hsv_image(image):
    """
    Saturate the pixels to the maximum.
    :param image: Image in RGB format.
    :return: Image in RGB format.
    """

    # Transform image from RGB to HSV.
    hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Split H, S and V channels.
    h, s, v = cv2.split(hsv_img)

    # Set all values of saturation channel to maximum (255).
    s_max = np.ones_like(s) * 255

    # Merge channels with saturation set to maximum.
    hsv_img_max_saturation = cv2.merge((h, s_max, v))

    # Converts the modified HSV image back into the BGR color space.
    bgr_img = cv2.cvtColor(hsv_img_max_saturation, cv2.COLOR_HSV2BGR)

    return bgr_img


def calculate_mean_colors(image):
    """
    Calculate the mean values of each channels in RGB format.
    :param image: Image in RGB format.
    :return: A matrix of tuple consists of the mean value of each channels.
    """
    # Divide image into 4x4 blocks.
    block_size = image.shape[0] // 4

    # Initialise result matrix.
    result_matrix = np.zeros((4, 4, 3), dtype=np.float32)

    for i in range(4):
        for j in range(4):
            # Focus on one block.
            block = image[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]

            # Calculate the mean value of each channels.
            mean_bgr = np.mean(block, axis=(0, 1))

            # Converts to a tuple and stores it in the result list
            result_matrix[i, j] = tuple(map(int, mean_bgr))

    return result_matrix.tolist()


def convert_to_color_labels(matrix):
    """
    Get the final classification results.
    :param matrix: A matrix of tuples. Tuples consist of means of each RGB channels.
    :return: The matrix of classification result.
    """

    def map_color(row):
        r, g, b = row
        if b > 150 and r < 150 and g < 150:
            return 'r'
        elif r > 150 and g < 150 and b < 150:
            return 'b'
        elif g > 150 and r < 150 and b < 150:
            return 'g'
        elif g > 150 and b > 150:
            return 'y'
        else:
            return 'w'

    # Apply conversion rules to each element.
    color_labels = [[map_color(pixel) for pixel in row] for row in matrix]
    return color_labels


def colorMatrix(image_path):
    """
    Function that automatically reads a colour pattern image from hard-disc and returns an array representing the colour pattern.
    :param image_path: The file name of image.
    :return: The matrix representing the color pattern.
    """
    # Get the coordinate of four black circles around the image.
    circles = getCircle(image_path)

    # Restored image to original shape.
    restored_img = transform(image_path, circles)

    # Increase pixel saturation.
    restored_img = saturate_hsv_image(restored_img)

    # Apply Gaussian Blur
    restored_img = cv2.GaussianBlur(restored_img, ksize=(5, 5), sigmaX=50)

    # Get the color pattern.
    res = calculate_mean_colors(restored_img)
    res = convert_to_color_labels(res)

    return res


if __name__ == "__main__":
    image_path = 'images1/noise_1.png'

    # Get the color matrix.
    result = colorMatrix(image_path)
    print(result)
