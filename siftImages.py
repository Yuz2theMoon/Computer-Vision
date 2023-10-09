import cv2 as cv
import sys
import numpy as np
import os


# Task 1
def display_task1(imagefile):
    # read image
    img = cv.imread(imagefile)

    if img is None:
        print(f"Failed to load image from {imagefile}")
        return
    # resize image
    img = cv.resize(img, (480, 600))
    # Convert to grayscale to get the Y component (luminance)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # create SIFT feature extractor
    # sift = cv.xfeatures2d.SIFT_create()
    sift = cv.SIFT_create()

    # detect features from the image
    keypoints, descriptors = sift.detectAndCompute(gray, None)

    # draw the detected key points
    #sift_image = cv.drawKeypoints(gray, keypoints, img)
    sift_image = cv.drawKeypoints(gray, keypoints, None,
                                  flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

     # Draw crosses "+" at the keypoints' locations
    for kp in keypoints:
        x, y = map(int, kp.pt)
        cv.drawMarker(sift_image, (x, y), color=(0, 0, 255), markerSize = 5, markerType=cv.MARKER_CROSS, thickness=1)
    
    # Combine original image and image with keypoints
    combined_image = cv.hconcat([img, sift_image])

    # show the image
    cv.imshow('image', combined_image)

    # save the image
    cv.imwrite("table-sift.jpg", combined_image)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # Print the number of detected keypoints
    print(f"# of SIFT keypoints in {imagefile} is: {len(keypoints)}")

# Task 2
def display_task2(imageList):
    keypointsList = []
    descriptorsList = []
    total_keypoints = 0
    for image_path_temp in imageList:
        image_temp = cv.imread(image_path_temp)
        image_temp_rescaled = rescale_image(image_temp, 600, 480)
        #convert to grayscale
        image = cv.cvtColor(image_temp_rescaled, cv.COLOR_BGR2GRAY)
        #init the SIFT detector
        sift = cv.SIFT_create()
        #detect keypoints and descriptors
        keypoints, descriptors = sift.detectAndCompute(image, None)
        #keypoints and descriptors
        keypointsList.append(keypoints)
        descriptorsList.append(descriptors)
        #total keypoints
        total_keypoints =total_keypoints + len(keypoints)
        #print keypoints for each image
        print(f"# of keypoints in {image_path_temp} is {len(keypoints)}")
    #matrices for K=5%, 10% and 20%
    print()
    k = [5, 10, 20]
    k_values = [int(total_keypoints * percentage / 100) for percentage in k]

    imageNameList = []
    for imgName in imageList:
        imageNameList.append(remove_file_extension(imgName))

    for k_value in k_values:
        print()
        print(f"K={k[k_values.index(k_value)]}%*(total number of keypoints)={k_value}")
        print()

        chi2_distances = cluster_and_compute_chi2(descriptorsList, k[k_values.index(k_value)]/100)
        
        #arrange the dissimilarity matrices in a readable format
        print("Dissimilarity Matrix")
        print(f"          {'     '.join(imageNameList)}")
        for i in range(len(imageList)):
            row = [f"{d:<10.2f}" if d != 0.00 else "          " for d in chi2_distances[i]]
            print(f"{remove_file_extension(imageList[i]):<10} {''.join(row)}")

#
def cluster_and_compute_chi2(images_sift_descriptors, k_percentage):
    # Cluster SIFT descriptors using K-means
    all_descriptors = np.vstack(images_sift_descriptors)
    total_descriptors = len(all_descriptors)
    K = int(total_descriptors * k_percentage)
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, _ = cv.kmeans(all_descriptors, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

    histograms = []
    for image_descriptors in images_sift_descriptors:
        #clusters
        cluster_labels = labels[:len(image_descriptors)]
        #histogram
        histogram, _ = np.histogram(cluster_labels, bins=range(K + 1), density=True)
        histograms.append(histogram)

    #Calculate χ² distance between histograms
    num_images = len(histograms)
    chi2_distances = np.zeros((num_images, num_images))
    for i in range(num_images):
        for j in range(num_images):
            chi2_distances[i, j] = calculate_chi2_distance(histograms[i], histograms[j])

    return chi2_distances

#function to calculate the χ² distance between the normalized histograms
def calculate_chi2_distance(hist1, hist2):
    return np.sum((hist1 - hist2)**2 / (hist1 + hist2 + 1e-10))

#function remove file extension
def remove_file_extension(filename):
    filename_without_extension, _ = os.path.splitext(filename)
    return filename_without_extension

#resize image public function
def rescale_image(image, target_w, target_h):
    #radio of width / height
    image_ratio = image.shape[1] / image.shape[0]
    #init new width and height
    res_w = int(target_h * image_ratio)
    res_h = target_h
    if image_ratio > target_w / target_h:
        res_w = target_w
        res_h = int(target_w / image_ratio)
    #resize image
    resized_image = cv.resize(image, (res_w, res_h))
    return resized_image

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python siftImages.py <imagefile1> [imagefile2 imagefile3 ...]")
    else:
        image_files = sys.argv[1:]
        if len(image_files) == 1:
            display_task1(image_files[0])
        else:
            display_task2(image_files[0:])