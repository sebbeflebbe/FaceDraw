#include <opencv2/opencv.hpp>
#include <SDL2/SDL.h>
#include <iostream>
#include <thread>
#include <cmath> // For sin and cos functions

void drawCircle(SDL_Renderer* renderer, int centerX, int centerY, int radius) {
    for (double angle = 0; angle <= 2 * M_PI; angle += 0.01) {
        int x = centerX + radius * cos(angle);
        int y = centerY + radius * sin(angle);
        SDL_RenderDrawPoint(renderer, x, y);
    }
}

void drawingThread(int* posX, int* posY, bool* running) {
    SDL_Init(SDL_INIT_VIDEO);
    SDL_Window* window = SDL_CreateWindow("Eye Drawing", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, 640, 480, SDL_WINDOW_SHOWN);
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);

    while (*running) {
        SDL_Event e;
        if (SDL_PollEvent(&e) && e.type == SDL_QUIT) {
            *running = false;
        }

        SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
        SDL_RenderClear(renderer);

        SDL_SetRenderDrawColor(renderer, 0, 0, 255, 255);
        drawCircle(renderer, *posX, *posY, 20);

        SDL_RenderPresent(renderer);
    }

    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
}

int main() {
    cv::VideoCapture cap(0);
    cv::CascadeClassifier face_cascade;
    face_cascade.load("/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml");

    int posX = 0, posY = 0;
    bool running = true;

    std::thread drawThread(drawingThread, &posX, &posY, &running);

    while (running) {
        cv::Mat frame, gray;
        cap >> frame;
        if (frame.empty()) break;

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        std::vector<cv::Rect> faces;
        face_cascade.detectMultiScale(gray, faces);

        for (const auto& face :faces){
            posX=face.x+face.width/2;
            posY=face.y+face.height/2;
        }
        cv::imshow("Eye Tracking", frame);
        if(cv::waitKey(1)=='q'){
            running=false;
        }
    }
    if(drawThread.joinable()){
        drawThread.join();
    }
    cap.release();
    cv::destroyAllWindows();
    return 0;
}
