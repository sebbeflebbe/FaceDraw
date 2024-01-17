#include <opencv2/opencv.hpp>
#include <SDL2/SDL.h>
#include <iostream>
#include <thread>
#include <atomic>
#include <vector>
#include <numeric>

const int WINDOW_WIDTH = 1280;
const int WINDOW_HEIGHT = 720;
const int SMOOTHING_FRAMES = 10;
const double SENSITIVITY_SCALE = 3.0; // Adjust this value for sensitivity

void drawCircle(SDL_Renderer* renderer, int centerX, int centerY, int radius) {
    for (double angle = 0; angle <= 2 * M_PI; angle += 0.01) {
        int x = centerX + radius * cos(angle);
        int y = centerY + radius * sin(angle);
        SDL_RenderDrawPoint(renderer, x, y);
    }
}

void drawingThread(std::atomic<int>* posX, std::atomic<int>* posY, std::atomic<bool>* running) {
    SDL_Init(SDL_INIT_VIDEO);
    SDL_Window* window = SDL_CreateWindow("Eye Drawing", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, WINDOW_WIDTH, WINDOW_HEIGHT, SDL_WINDOW_SHOWN);
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);

    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255); // White background
    SDL_RenderClear(renderer);
    SDL_RenderPresent(renderer);

    int lastX = WINDOW_WIDTH / 2, lastY = WINDOW_HEIGHT / 2; // Start from the center

    while (running->load()) {
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) {
                running->store(false);
            }
        }

        int currentX = posX->load();
        int currentY = posY->load();

        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255); // Black color for the line
        SDL_RenderDrawLine(renderer, lastX, lastY, currentX, currentY);

        lastX = currentX;
        lastY = currentY;

        SDL_RenderPresent(renderer);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
}

int main() {
    cv::VideoCapture cap(0);
    cv::CascadeClassifier face_cascade;
    if (!face_cascade.load("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml")) {
        std::cerr << "Error loading cascade classifier." << std::endl;
        return -1;
    }

    std::atomic<int> posX(WINDOW_WIDTH / 2), posY(WINDOW_HEIGHT / 2);
    std::atomic<bool> running(true);
    bool centerSet = false;
    cv::Point2i initialCenter;

    std::thread drawThread(drawingThread, &posX, &posY, &running);

    while (running.load()) {
        cv::Mat frame, gray;
        cap >> frame;
        if (frame.empty()) break;

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        std::vector<cv::Rect> faces;
        face_cascade.detectMultiScale(gray, faces);

        if (!faces.empty()) {
            auto largest_face = *std::max_element(faces.begin(), faces.end(), [](const cv::Rect& a, const cv::Rect& b) {
                return a.area() < b.area();
            });

            cv::Point2i faceCenter(largest_face.x + largest_face.width / 2, largest_face.y + largest_face.height / 2);

            if (!centerSet) {
                initialCenter = faceCenter;
                centerSet = true;
            }

            int relativeX = (faceCenter.x - initialCenter.x) * SENSITIVITY_SCALE + WINDOW_WIDTH / 2;
            int relativeY = (faceCenter.y - initialCenter.y) * SENSITIVITY_SCALE + WINDOW_HEIGHT / 2;

            posX.store(WINDOW_WIDTH - relativeX); // Mirroring the X coordinate
            posY.store(relativeY);
        }

        cv::imshow("Eye Tracking", frame);
        if (cv::waitKey(1) == 'q') {
            running.store(false);
        }
    }

    if (drawThread.joinable()) {
        drawThread.join();
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
