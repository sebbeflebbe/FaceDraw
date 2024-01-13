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

void eyeDetectionThread(cv::CascadeClassifier& eye_cascade, const int framesThreshold, cv::VideoCapture& cap) {
    int framesWithoutEyes = 5;
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

        std::vector<cv::Rect> eyes;
        eye_cascade.detectMultiScale(gray, eyes, 1.25, 12);

        for(const auto& eye : eyes) {
            cv::rectangle(frame, eye, cv::Scalar(0, 255, 0), 2);
        }

        if (eyes.empty()) {
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
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Error opening video stream" << std::endl;
        return -1;
    }

    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);

    cv::CascadeClassifier eye_cascade;
    if (!eye_cascade.load("/usr/share/opencv4/haarcascades/haarcascade_eye.xml")) {
        std::cerr << "Error loading eye cascade" << std::endl;
        return -1;
    }

    const int framesThreshold = 10;
    std::thread soundThread(playSound, "./alarm.mp3");
    std::thread eyeDetectionWorker(eyeDetectionThread, std::ref(eye_cascade), framesThreshold, std::ref(cap));

    if (eyeDetectionWorker.joinable()) {
        eyeDetectionWorker.join();
    }
    if (soundThread.joinable()) {
        soundThread.join();
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
