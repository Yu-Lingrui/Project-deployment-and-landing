#include <iostream>
#include <vector>

std::vector<std::vector<int>> get_slice_bboxes(
    int image_height,
    int image_width,
    int slice_height = 0,
    int slice_width = 0,
    double overlap_height_ratio = 0.2,
    double overlap_width_ratio = 0.2
) {
    std::vector<std::vector<int>> slice_bboxes;
    int y_max = 0, y_min = 0;

    if (slice_height && slice_width) {
        int y_overlap = int(overlap_height_ratio * slice_height);
        int x_overlap = int(overlap_width_ratio * slice_width);
    } else {
        throw std::runtime_error("Compute type is not auto and slice width and height are not provided.");
    }

    while (y_max < image_height) {
        int x_min = 0, x_max = 0;
        y_max = y_min + slice_height;
        while (x_max < image_width) {
            int x_max = x_min + slice_width;
            if (y_max > image_height || x_max > image_width) {
                int xmax = std::min(image_width, x_max);
                int ymax = std::min(image_height, y_max);
                int xmin = std::max(0, xmax - slice_width);
                int ymin = std::max(0, ymax - slice_height);
                slice_bboxes.push_back({xmin, ymin, xmax, ymax});
            } else {
                slice_bboxes.push_back({x_min, y_min, x_max, y_max});
            }
            x_min = x_max - x_overlap;
        }
        y_min = y_max - y_overlap;
    }
    return slice_bboxes;
}

int main() {
    try {
        std::vector<std::vector<int>> bboxes = get_slice_bboxes(image_height / 2, image_weight, image_height / 4,
                                                image_weight / 2, 0.2, 0.2);
        for (const auto& bbox : bboxes) {
            std::cout << "[" << bbox[0] << ", " << bbox[1] << ", " << bbox[2] << ", " << bbox[3] << "]" << std::endl;
        }
    } catch (const std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
    }
    return 0;
}

#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>

using namespace std;
using namespace cv;

int main() {
    // Load the image
    Mat image_pil_arr = imread("image.jpg", IMREAD_COLOR);

    // Define the slice bounding boxes
    vector<std::vector<int>> bboxes = {std::vector<int>{0, 0, 100, 100}, std::vector<int>{100, 0, 200, 100}};

    int n_ims = 0;

    // Iterate over the slice bounding boxes
    for (const auto& bbox : bboxes) {
        n_ims++;

        // Extract the slice
        Rect tlx_brx(bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]);
        Mat image_pil_slice = image_pil_arr(tlx_brx);

        // Resize the image
        cv::Mat resized_image = cv::resize(image_pil_slice, (image_height, image_width), cv::INTER_LINEAR);

        // Display the slice
        imshow("Slice " + to_string(n_ims), image_pil_slice);
        waitKey(0);
    }

    return 0;
}
