#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // Initialize video capture on webcam
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video stream" << std::endl;
        return -1;
    }

    // Load the Haar Cascade for face detection
    cv::CascadeClassifier face_cascade;
    if (!face_cascade.load("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml")) {
    std::cerr << "Error loading face cascade" << std::endl;
    return -1;
}


    // Main loop to capture and process frames
    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) {
            break;
        }

        // Convert to grayscale for face detection
        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::equalizeHist(gray, gray);

        // Detect faces
        std::vector<cv::Rect> faces;
        face_cascade.detectMultiScale(gray, faces);

        // Draw rectangles around detected faces
        for (const auto& face : faces) {
            cv::rectangle(frame, face, cv::Scalar(255, 0, 0), 2);
        }

        // Show the frame with detected faces
        cv::imshow("Face Detection", frame);

        // Break loop on 'q' key press
        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    // Release resources and close windows
    cap.release();
    cv::destroyAllWindows();
    return 0;
}
