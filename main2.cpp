#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <chrono>

std::atomic<bool> finished(false);
std::atomic<bool> playAlert(false);

void playSound(const std::string& soundFile) {
    while(!finished.load()) {
        if(playAlert.load()) {
            std::string command = "mpg123 " + soundFile;
            system(command.c_str());
        }
        // Sleep briefly to prevent tight loop when not playing
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
}

void eyeDetectionThread(cv::CascadeClassifier& eye_cascade, cv::CascadeClassifier& face_cascade, const int framesThreshold, cv::VideoCapture& cap) {
    int framesWithoutEyes = 0;
    cv::Mat frame;

    while (!finished.load()) {
        cap >> frame;
        if (frame.empty()) {
            finished.store(true);
            break;
        }

        cv::Mat gray;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        cv::equalizeHist(gray, gray);

        // Detect faces
        std::vector<cv::Rect> faces;
        face_cascade.detectMultiScale(gray, faces);

        bool eyesDetected = false;
        for (const auto &face : faces) {
    double scaleFactor = static_cast<double>(face.width) / 200.0; // Tune this base value
    int minEyeSize = static_cast<int>(20 * scaleFactor); // Tune this base value
    int maxEyeSize = static_cast<int>(40 * scaleFactor); // Add a max size if needed

    cv::Mat faceROI = gray(face);
    std::vector<cv::Rect> eyes;
    eye_cascade.detectMultiScale(faceROI, eyes, 1.25, 12, 0, cv::Size(minEyeSize, minEyeSize), cv::Size(maxEyeSize, maxEyeSize));

            for (const auto &eye : eyes) {
                eyesDetected = true;
                cv::Rect eye_rect(face.x + eye.x, face.y + eye.y, eye.width, eye.height);
                cv::rectangle(frame, eye_rect, cv::Scalar(0, 255, 0), 2);
            }
        }

        if (!eyesDetected) {
            framesWithoutEyes++;
            if (framesWithoutEyes >= framesThreshold) {
                playAlert.store(true);
            }
        } else {
            framesWithoutEyes = 0;
            playAlert.store(false);
        }

        cv::imshow("Driver Drowsiness Detection", frame);
        if (cv::waitKey(1) == 'q') {
            finished.store(true);
            break;
        }
    }
}


int main() {
    // Initialize VideoCapture and CascadeClassifiers
    cv::VideoCapture cap(0);
    cv::CascadeClassifier eye_cascade, face_cascade;

    // Check if video stream opened successfully
    if (!cap.isOpened()) {
        std::cerr << "Error opening video stream" << std::endl;
        return -1;
    }

    // Set camera properties
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    // Load the eye and face cascade classifiers
    if (!eye_cascade.load("/usr/share/opencv4/haarcascades/haarcascade_eye.xml")) {
        std::cerr << "Error loading eye cascade" << std::endl;
        return -1;
    }
    if (!face_cascade.load("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml")) {
        std::cerr << "Error loading face cascade" << std::endl;
        return -1;
    }

    // Threshold for eye detection
    const int framesThreshold = 10;

    // Start threads for sound playback and eye detection
    std::thread soundThread(playSound, "./alarm.mp3");
    std::thread eyeDetectionWorker(eyeDetectionThread, std::ref(eye_cascade), std::ref(face_cascade), framesThreshold, std::ref(cap));

    // Wait for threads to finish
    if (eyeDetectionWorker.joinable()) {
        eyeDetectionWorker.join();
    }
    if (soundThread.joinable()) {
        soundThread.join();
    }

    // Release resources
    cap.release();
    cv::destroyAllWindows();
    return 0;
}

