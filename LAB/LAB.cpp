#include "pch.h"

#include "opencv2/opencv.hpp"

#include <iostream>
#include <sstream>

using namespace cv;
using namespace std;

int main(int argc, char** argv)
{   
    //dad
    Mat frame;
    Mat background;
    Mat object;

    string videoPath = "C:\\Users\\User\\source\\repos\\LAB\\Vidandpicture\\vtest.avi";
    VideoCapture cap(videoPath); // open the default camera
    /*VideoCapture cap(argv[1]); */
    if (!cap.isOpened())  // check if we succeeded
        return -1;

    cap.read(frame);
    Mat acc = Mat::zeros(frame.size(), CV_32FC1);

    namedWindow("Video");
    namedWindow("Frame");
    namedWindow("Background");
    namedWindow("Foreground");

    for (;;)
    {
        Mat gray;
        cap >> frame;
        imshow("Video", frame);

        cvtColor(frame, gray, COLOR_BGR2GRAY);

        // running average
        // B = alpha * I + (1-alpha) * B;
        float alpha = 0.05;
        accumulateWeighted(gray, acc, alpha);

        // scale to 8-bit unsigned
        convertScaleAbs(acc, background);

        imshow("Background", background);

        // background subtraction
        // O = | I - B |
        subtract(background, gray, object);
        imshow("Frame", object);

        threshold(object, object, 25, 255, 0);

        imshow("Foreground", object);

        //imshow("Threshold", gray);
        if (waitKey(30) >= 0) break;
    }
    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}