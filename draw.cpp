#include <opencv2/opencv.hpp>
#include <SDL2/SDL.h>
#include <iostream>
#include <thread>
#include <atomic>

void drawCircle(SDL_Renderer* renderer, int centerX, int centerY, int radius) {
    for (double angle = 0; angle <= 2 * M_PI; angle += 0.01) {
        int x = centerX + radius * cos(angle);
        int y = centerY + radius * sin(angle);
        SDL_RenderDrawPoint(renderer, x, y);
    }
}

void drawingThread(std::atomic<int>* posX, std::atomic<int>* posY, std::atomic<bool>* running) {
    SDL_Init(SDL_INIT_VIDEO);
    SDL_Window* window = SDL_CreateWindow("Eye Drawing", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, 640, 480, SDL_WINDOW_SHOWN);
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);

    // Set the background color to white once
    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255); // White background
    SDL_RenderClear(renderer);

    int lastX = 320, lastY = 240; // Start from the center

    while (running->load()) {
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) {
                running->store(false);
            }
        }

        // Retrieve current eye positions
        int currentX = posX->load();
        int currentY = posY->load();

        // Draw a black line from the last position to the current position
        SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255); // Black color for the line
        SDL_RenderDrawLine(renderer, lastX, lastY, currentX, currentY);

        // Update last positions
        lastX = currentX;
        lastY = currentY;

        // Update the screen with any rendering performed since the previous call
        SDL_RenderPresent(renderer);

        // Introduce a small delay to prevent too rapid drawing
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
}


int main() {
    cv::VideoCapture cap(0);
    cv::CascadeClassifier face_cascade;
    face_cascade.load("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml");

    std::atomic<int> posX(320), posY(240); // Initialize to center of the window
    std::atomic<bool> running(true);

    std::thread drawThread(drawingThread, &posX, &posY, &running);

    while (running.load()) {
        cv::Mat frame, gray;
        cap >> frame;
        if (frame.empty()) break;

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        std::vector<cv::Rect> faces;
        face_cascade.detectMultiScale(gray, faces);

        for (const auto& face : faces) {
            posX.store(face.x + face.width / 2);
            posY.store(face.y + face.height / 2);
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
