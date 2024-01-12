#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cstdlib> // for system command

int main() {
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video stream" << std::endl;
        return -1;
    }

    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    cv::CascadeClassifier face_cascade, eye_cascade;
    if (!face_cascade.load("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml")) {
        std::cerr << "Error loading face cascade" << std::endl;
        return -1;
    }
    if (!eye_cascade.load("/usr/share/opencv4/haarcascades/haarcascade_eye.xml")) {
        std::cerr << "Error loading eye cascade" << std::endl;
        return -1;
    }

    int framesWithoutEyes = 0;
    const int framesThreshold = 5; // Number of consecutive frames without eye detection to trigger alert
    const char* alertSound = "mpg123 ./alarm.mp3"; // Replace with the path to your alert sound file

    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::equalizeHist(gray, gray);

        std::vector<cv::Rect> faces;
        face_cascade.detectMultiScale(gray, faces);

        bool eyesDetected = false;
        for (const auto& face : faces) {
            cv::Mat faceROI = gray(face);
            std::vector<cv::Rect> eyes;
            eye_cascade.detectMultiScale(faceROI, eyes, 1.25, 15);

            if (!eyes.empty()) {
                eyesDetected = true;
                framesWithoutEyes = 0; // Reset the counter as eyes are detected

                // Draw rectangles around the eyes for visualization
                for (const auto& eye : eyes) {
                    cv::Rect eye_rect(face.x + eye.x, face.y + eye.y, eye.width, eye.height);
                    cv::rectangle(frame, eye_rect, cv::Scalar(0, 255, 0), 2);
                }
            }

            // Draw rectangle around the face
            cv::rectangle(frame, face, cv::Scalar(255, 0, 0), 2);
        }

        if (!eyesDetected) {
            framesWithoutEyes++;
            if (framesWithoutEyes >= framesThreshold) {
                std::cout << "Drowsiness detected!" << std::endl;
                system(alertSound); // Play sound alert
                framesWithoutEyes = 0; // Reset counter to avoid continuous sound playing
            }
        }

        cv::imshow("Driver Drowsiness Detection", frame);
        if (cv::waitKey(1) == 'q') {
            break;
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
