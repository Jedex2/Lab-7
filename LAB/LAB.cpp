#include "pch.h"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc.hpp"

using namespace cv;
using namespace std;
//www

int threshold_value = 45;
int threshold_type = 3;
int const max_value = 255;
int const max_type = 4;
int const max_BINARY_value = 255;

Mat src, src_gray, dst, grad, grad_x, grad_y, abs_grad_x, abs_grad_y, originalImage;
int scale = 1;
int delta = 0;
int ddepth = CV_16S;

RNG rng4(12345);

void Threshold_Demo(int, void*);
bool contour_features(vector<Point> ct);
Rect largestBoundingBox;

void Threshold_Demo(int, void*) {
    threshold(grad, dst, 20, 255, 1);

    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(dst, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));

    vector<vector<Point>> contours_poly(contours.size());
    vector<vector<Point>> contours_cycle;
    vector<Rect> boundRect(contours.size());

    int j = 0;
    largestBoundingBox = Rect();
    double maxArea = 0;

    for (int i = 0; i < contours.size(); i++) {
        approxPolyDP(Mat(contours[i]), contours_poly[j], 1, true);
        if (contours_poly[j].size() > 5) {
            if (contour_features(contours_poly[j])) {
                contours_cycle.push_back(contours_poly[j]);
                boundRect[j] = boundingRect(Mat(contours_poly[j]));
                double area = boundRect[j].area();
                if (area > maxArea) {
                    maxArea = area;
                    largestBoundingBox = boundRect[j];
                }
            }
            j++;
        }
    }

    Mat drawing = Mat::zeros(dst.size(), CV_8UC3);
    for (int i = 0; i < contours_cycle.size(); i++) {
        Scalar color = Scalar(rng4.uniform(0, 255), rng4.uniform(0, 255), rng4.uniform(0, 255));
        rectangle(drawing, boundRect[i].tl(), boundRect[i].br(), color, 2, 8, 0);
    }

    if (largestBoundingBox.area() > 0) {
        rectangle(drawing, largestBoundingBox.tl(), largestBoundingBox.br(), Scalar(0, 0, 255), 3, 8, 0);
    }

    /*imshow("Contours", drawing);*/
}

bool contour_features(vector<Point> ct) {
    double area = contourArea(ct);
    Moments mu = moments(ct);
    double hu[7];
    HuMoments(mu, hu);

    return hu[0] < 0.18;
}

int main() {
    const char* imagePath = "C:\\forvsc++\\NewC++\\Good (6).jpg";

    originalImage = imread(imagePath);
    if (originalImage.empty()) {
       
        return -1;
    }


    rotate(originalImage, originalImage, ROTATE_180);
    /*imshow("Original Image", originalImage);*/

    cvtColor(originalImage, src_gray, COLOR_BGR2GRAY);
    /*imshow("Grayscale Image", src_gray);*/

    Mat blurredImage;
    GaussianBlur(src_gray, blurredImage, Size(5, 5), 0, 0);
    /*imshow("Blurred Image", blurredImage);*/

    Sobel(blurredImage, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
    Sobel(blurredImage, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
    convertScaleAbs(grad_x, abs_grad_x);
    convertScaleAbs(grad_y, abs_grad_y);
    addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);

    /*imshow("Sobel Edge Detection", grad);*/

    Threshold_Demo(0, 0);



    if (largestBoundingBox.area() > 0) {
        Mat croppedImage = originalImage(largestBoundingBox).clone();
        imshow("Cropped Image", croppedImage);
        imwrite("Cropped_Output.jpg", croppedImage);
       
        // ทำ Polar Transform
        Mat polarImage;
        Point2f center(croppedImage.cols / 2.0f, croppedImage.rows / 2.0f);
        double maxRadius = min(center.x, center.y);

        // ใช้ linearPolar หรือ logPolar
        linearPolar(croppedImage, polarImage, center, maxRadius, INTER_LINEAR);
        // logPolar(croppedImage, polarImage, center, maxRadius, INTER_LINEAR);

        /*namedWindow("Polar Transform", WINDOW_AUTOSIZE);
        imshow("Polar Transform", polarImage);*/
        imwrite("Polar_Output.jpg", polarImage);
       

        // ใช้ cropRegion แทนการกำหนดค่า ROI ตรงๆ
        Rect cropRegion(polarImage.cols - 50, 0, 50, polarImage.rows);
        Mat croppedRight = polarImage(cropRegion).clone();
        /*namedWindow("Cropped Right Polar", WINDOW_AUTOSIZE);
        imshow("Cropped Right Polar", croppedRight);*/
        imwrite("Cropped_Right_Polar.jpg", croppedRight);
        

        Mat rotatedCroppedRight;
        rotate(croppedRight, rotatedCroppedRight, ROTATE_90_CLOCKWISE);
        namedWindow("Rotated Cropped Right Polar", WINDOW_AUTOSIZE);
        imshow("Rotated Cropped Right Polar", rotatedCroppedRight);
        imwrite("Rotated_Cropped_Right_Polar.jpg", rotatedCroppedRight);
        

        cvtColor(rotatedCroppedRight, rotatedCroppedRight, COLOR_BGR2GRAY);
        int hval[1000] = { 0 };
        int max_val = 0;
        for (int i = 0; i < rotatedCroppedRight.cols; i++)
        {
            int sum = 0;
            for (int j = 0; j < rotatedCroppedRight.rows; j++)
            {
                sum += rotatedCroppedRight.at<unsigned char>(j, i);
            }
            hval[i] = sum;
            if (sum > max_val) max_val = sum;
        }

        Mat imgHist = Mat::zeros(Size(1000, 1000), rotatedCroppedRight.type());
        for (int i = 0; i < 1000; i++)
        {
            hval[i] = hval[i] / (float)max_val * 100;
            line(imgHist, Point(i, 100 - hval[i]), Point(i, 100), Scalar(255, 255, 255));
            if (hval[i] == 0)
                line(imgHist, Point(i, 0), Point(i, 100), Scalar(255, 255, 255));
        }
        imshow("Histogram", imgHist);

        int left = 0, right = rotatedCroppedRight.cols - 1;
        int thresholdLow = 40; // ค่า threshold สำหรับพิจารณาว่าพิกเซลในคอลัมน์นั้นเป็นข้อมูลที่ไม่สำคัญ
        int thresholdHigh = 60;
        // หาตำแหน่งขอบซ้ายที่มี hval[i] สูงกว่า threshold
        for (int i = 0; i < rotatedCroppedRight.cols; i++) {
            if (hval[i] > thresholdLow && hval[i] < thresholdHigh) {  // ถ้า hval[i] มากกว่าค่าที่กำหนด
                left = i;
                break;  // หาขอบซ้ายที่มีค่า hval[i] มากกว่า threshold
            }
        }

        // หาตำแหน่งขอบขวาที่มี hval[i] สูงกว่า threshold
        for (int i = rotatedCroppedRight.cols - 1; i >= 0; i--) {
            if (hval[i] > thresholdLow && hval[i] < thresholdHigh) {  // ถ้า hval[i] มากกว่าค่าที่กำหนด
                right = i;
                break;  // หาขอบขวาที่มีค่า hval[i] มากกว่า threshold
            }
        }
        if (right > left) { // ถ้าพบขอบเขตที่ใช้งานได้
            Rect cropRegion(left, 0, right - left, rotatedCroppedRight.rows); // กำหนดขอบเขตการครอบ
            Mat finalCropped = rotatedCroppedRight(cropRegion).clone(); // ครอบภาพตามขอบเขตที่กำหนด

            imshow("last Image", finalCropped); // แสดงภาพที่ถูกครอบ
            imwrite("Cropped_Image.jpg", finalCropped); // บันทึกภาพที่ครอบ
            
        }

        waitKey(0); // Wait for a keystroke in the window
        return 0;
    }

}