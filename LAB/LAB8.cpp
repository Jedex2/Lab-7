#include "pch.h"
#include "opencv2/opencv.hpp"
#include "opencv2/objdetect.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/video/tracking.hpp"
#include <iostream>

using namespace cv;
using namespace std;
RNG rng4(12345);

class NPlate // class representing a person
{
public:
    int life;
    Rect boundRect;
    Point2f center;
    Rect boundRect_pre;
    Point2f center_pre;
    Scalar color;
    vector<Point2f> tar;
    vector<Point2f>tar_pre;
    KalmanFilter KF;
    NPlate() : KF(4, 2, 0) // Initialize the member KalmanFilter
    {
        KF.transitionMatrix = (Mat_<float>(4, 4) << 1, 0, 1, 0,
            0, 1, 0, 1,
            0, 0, 1, 0,
            0, 0, 0, 1);
        setIdentity(KF.measurementMatrix);
        setIdentity(KF.processNoiseCov, Scalar::all(1e-4));
        setIdentity(KF.measurementNoiseCov, Scalar::all(1));
        setIdentity(KF.errorCovPost, Scalar::all(1e-4));
        life = 5;
    }

};

vector<NPlate> nObjs; // tracking people in the video

void delete_people(int idx)
{
    nObjs.erase(nObjs.begin() + idx);
}

float cal_dist(NPlate blob, NPlate obj)
{
    return sqrt(pow(blob.center.x - obj.center.x, 2) + pow(blob.center.y - obj.center.y, 2));
}

void update_the_people(int idx, NPlate blob)
{
    nObjs[idx].life = 5;

    nObjs[idx].tar.push_back(nObjs[idx].center);
    nObjs[idx].center = blob.center;
    nObjs[idx].boundRect = blob.boundRect;

    if (nObjs[idx].tar_pre.size() == 0)
    {
        nObjs[idx].KF.statePre.at<float>(0) = blob.center.x;
        nObjs[idx].KF.statePre.at<float>(1) = blob.center.y;
        nObjs[idx].KF.statePre.at<float>(2) = 0; //vx
        nObjs[idx].KF.statePre.at<float>(3) = 0; //vy

        nObjs[idx].KF.statePost.at<float>(0) = blob.center.x;
        nObjs[idx].KF.statePost.at<float>(1) = blob.center.y;
        nObjs[idx].KF.statePost.at<float>(2) = 0; //vx
        nObjs[idx].KF.statePost.at<float>(3) = 0; //vy

        nObjs[idx].center_pre.x = blob.center.x;
        nObjs[idx].center_pre.y = blob.center.y;
        nObjs[idx].tar_pre.push_back(nObjs[idx].center_pre);
    }
    else
    {
        Mat prediction = nObjs[idx].KF.predict();
        nObjs[idx].center_pre.x = prediction.at<float>(0);
        nObjs[idx].center_pre.y = prediction.at<float>(1);

        Mat_<float> measurement(2, 1); measurement.setTo(Scalar(0));
        measurement(0) = blob.center.x;
        measurement(1) = blob.center.y;

        Mat estimated = nObjs[idx].KF.correct(measurement);
        nObjs[idx].center_pre.x = estimated.at<float>(0);
        nObjs[idx].center_pre.y = estimated.at<float>(1);
        nObjs[idx].tar_pre.push_back(nObjs[idx].center_pre);
    }
}

void add_people(NPlate blob)
{
    blob.KF.statePre.at<float>(0) = blob.center.x;
    blob.KF.statePre.at<float>(1) = blob.center.y;
    blob.KF.statePre.at<float>(2) = 0; //vx
    blob.KF.statePre.at<float>(3) = 0; //vy
    nObjs.push_back(blob);
}

int main(int argc, char** argv)
{
    string videoPath = "C:\\Users\\User\\source\\repos\\Lab-8\\x64\\Debug\\lpr1.mp4";
    VideoCapture cap(videoPath);
    if (!cap.isOpened())
        return -1;

    String lpr_cascade_name = "C:\\Users\\User\\Downloads\\cascade.xml";
    CascadeClassifier lpr_cascade;

    if (!lpr_cascade.load(lpr_cascade_name))
    {
        printf("--(!)Error loading\n");
        return -1;
    };

    Mat imgLpr;

    for (;;)
    {
        Mat frame;
        cap >> frame;

        Mat imgFrame;
        Mat croppedRoi(frame, Rect(0, frame.rows / 3, frame.cols / 2, frame.rows / 3));
        croppedRoi.copyTo(imgFrame);

        std::vector<Rect> lprs;
        Mat frame_gray;

        cvtColor(imgFrame, frame_gray, COLOR_BGR2GRAY);
        equalizeHist(frame_gray, frame_gray);

        lpr_cascade.detectMultiScale(frame_gray, lprs, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

        vector<NPlate> nBlobs;

        for (size_t i = 0; i < lprs.size(); i++)
        {
            NPlate nBlob;
            nBlob.boundRect = lprs[i];
            nBlob.center = Point2f(lprs[i].x + lprs[i].width / 2, lprs[i].y + lprs[i].height / 2);
            nBlob.color = Scalar(rng4.uniform(0, 255), rng4.uniform(0, 255), rng4.uniform(0, 255));
            nBlobs.push_back(nBlob);
        }

        for (int i = 0; i < nBlobs.size(); i++)
        {
            float dist_min = 10000;
            int dist_idx = -1;
            for (int j = 0; j < nObjs.size(); j++) {

                float dist = cal_dist(nBlobs[i], nObjs[j]);
                if (dist < dist_min)
                {
                    dist_idx = j;
                    dist_min = dist;
                }
            }
            if (dist_idx >= 0)
            {
                int TH = 30;
                if (dist_min < TH)
                    update_the_people(dist_idx, nBlobs[i]);
                else
                    add_people(nBlobs[i]);
            }
            else
            {
                add_people(nBlobs[i]);
            }
        }

        for (int i = 0; i < nObjs.size(); i++)
        {
            nObjs[i].life--;
            if (nObjs[i].life < 0)
            {
                delete_people(i);
            }
        }

        for (int i = 0; i < nObjs.size(); i++)
        {
            if (nObjs[i].tar.size() > 5 && nObjs[i].center.x > 50)
            {
                Mat croppedRoi(imgFrame, nObjs[i].boundRect);
                croppedRoi.copyTo(imgLpr);
                resize(imgLpr, imgLpr, Size(imgLpr.cols * 4, imgLpr.rows * 4));
                rectangle(imgFrame, nObjs[i].boundRect.tl(), nObjs[i].boundRect.br(), nObjs[i].color, 2, 8, 0);
                if (nObjs[i].tar.size() > 2)
                    for (int j = 0; j < nObjs[i].tar.size() - 1; j++)
                    {
                        line(imgFrame, nObjs[i].tar[j], nObjs[i].tar[j + 1], nObjs[i].color, 1, 8, 0);
                    }
                if (nObjs[i].tar_pre.size() > 3)
                    for (int j = 1; j < nObjs[i].tar_pre.size() - 1; j++)
                    {
                        line(imgFrame, nObjs[i].tar_pre[j], nObjs[i].tar_pre[j + 1], nObjs[i].color, 3, 8, 0);
                    }
            }
            if (imgLpr.cols > 0)
            {
                Rect t_rect = Rect(0, 0, imgLpr.cols, imgLpr.rows);
                imgLpr.copyTo(imgFrame(t_rect));
            }
        }
        imshow("Frame", imgFrame);

        if (waitKey(30) >= 0) break;
    }

    return 0;
}