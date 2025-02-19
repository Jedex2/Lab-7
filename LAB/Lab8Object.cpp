#include "pch.h"

#include "opencv2/opencv.hpp"

#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

#include <iostream>
#include <stdio.h>

using namespace std;
using namespace cv;

/** Function Headers */
void detectAndDisplay1(Mat frame);

/** Global variables */
String face_cascade_name1 = "C:\\Users\\User\\Downloads\\LabHAARbin\\LabHAARbin\\haarcascade\\cascade.xml";
//String face_cascade_name1 = "C:\\Users\\User\\source\\repos\\LAB 8\\cascade (1).xml";
String eyes_cascade_name1 = "C:\\Users\\User\\source\\repos\\LAB 8\\haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade1;
CascadeClassifier eyes_cascade1;
String window_name1 = "Capture - Face detection";

/** @function main */
int main_object_detect(void)
{
    VideoCapture capture;
    Mat frame;

    //-- 1. Load the cascades
    if (!face_cascade1.load(face_cascade_name1)) { printf("--(!)Error loading face cascade\n"); return -1; };
    if (!eyes_cascade1.load(eyes_cascade_name1)) { printf("--(!)Error loading eyes cascade\n"); return -1; };

    //-- 2. Read the video stream
    //capture.open(0);
    ///if ( ! capture.isOpened() ) { printf("--(!)Error opening video capture\n"); return -1; }

    FILE* fp;
    fp = fopen("test.dat", "r");
    if (fp == NULL) return 0;

    //while (  capture.read(frame) )
    do
    {
        char name[100];
        fscanf(fp, "%s", name);
        frame = imread(name);
        if (frame.empty())
        {
            printf(" --(!) No captured frame -- Break!");
            break;
        }
        Size size(400, 800);
        resize(frame, frame, size);

        //-- 3. Apply the classifier to the frame
        detectAndDisplay1(frame);

        int c = waitKey(1000);
        if ((char)c == 27) { break; } // escape
    } while (1);
    return 0;
}

/** @function detectAndDisplay */
void detectAndDisplay1(Mat frame)
{
    std::vector<Rect> faces;
    Mat frame_gray;

    cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
    equalizeHist(frame_gray, frame_gray);

    //-- Detect faces
    face_cascade1.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

    for (size_t i = 0; i < faces.size(); i++)
    {
        rectangle(frame, Point(faces[i].x, faces[i].y),
            Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height),
            Scalar(0, 255, 0), 1, 1);
        /*
        Mat faceROI = frame_gray( faces[i] );
        std::vector<Rect> eyes;

        //-- In each face, detect eyes
        eyes_cascade1.detectMultiScale( faceROI, eyes, 1.1, 2, 0 |CASCADE_SCALE_IMAGE, Size(30, 30) );

        for( size_t j = 0; j < eyes.size(); j++ )
        {
            Point eye_center( faces[i].x + eyes[j].x + eyes[j].width/2, faces[i].y + eyes[j].y + eyes[j].height/2 );
            int radius = cvRound( (eyes[j].width + eyes[j].height)*0.25 );
            circle( frame, eye_center, radius, Scalar( 255, 0, 0 ), 4, 8, 0 );
        }
        */
    }
    //-- Show what you got
    imshow(window_name1, frame);
}