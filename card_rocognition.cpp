#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>
#include<tesseract/baseapi.h>
#include<fstream>
#include<vector>
#include<string>

using namespace std;
using namespace cv;
using namespace tesseract;

Mat src;

char window_in[] = "input image";
char window_out[] = "output image";
int closeTimes = 1;

Mat binarize(Mat input);
RotatedRect GetLocatedArea(Mat input);
Mat getNumbers(RotatedRect box);
vector<Mat>cutNumbers(Mat input);
string numberIdentify(vector<Mat>input);

TessBaseAPI ocr;

int main(int argc, char** argv) {
    src = imread("F:/ps/cp11.jpg");
	if (!src.data) {
		printf("could not load image...\n");
		return -1;
	}

	namedWindow(window_in, WINDOW_AUTOSIZE);
	//namedWindow(window_out, WINDOW_AUTOSIZE);
	float value = static_cast<float> (src.cols) / static_cast<float>(src.rows);
	resize(src, src, Size(450 * value, 450), 0, 0);
	imshow(window_in, src);

	//得到闭图像
	int loopTimes = 0;
begin:
	Mat closeImg = binarize(src);

	//车牌定位
	RotatedRect region = GetLocatedArea(closeImg);
	Mat locatedImg;
	
	//提取车牌
	Mat numberImg = getNumbers(region);

	vector<vector<Point>>numberContours;
	findContours(numberImg, numberContours, RETR_TREE, CHAIN_APPROX_SIMPLE);
	vector<vector<Point>>contours_poly(numberContours.size());
	Rect testRect;
	int count = 0;
	for (int i = 0; i < numberContours.size(); i++) {
		approxPolyDP(numberContours[i], contours_poly[i], 3, true);
		testRect = boundingRect(contours_poly[i]);
		if (testRect.area() < 70 * 150 && testRect.area() > 1500) {
			count++;
		}
	}
	if (count < 8 && loopTimes < 8) {
		closeTimes++;
		if (loopTimes == 7) {
			closeTimes = 4;
		}
		loopTimes++;
		goto begin;
	}

	src.copyTo(locatedImg);
	Point2f pts[4];
	region.points(pts);
	for (int i = 0; i < 4; i++) {
		line(locatedImg, pts[i], pts[(i + 1) % 4], Scalar(0, 255, 0), 2, LINE_AA);
	}
	imshow("located", locatedImg);

	//imshow("number", numberImg);
	//cout << count << endl;
	//字符分割
	vector<Mat>cutImg = cutNumbers(numberImg);

	//模板匹配
	//string number = numberIdentify(cutImg);

	//cout << number <<"\n"<< endl;

	/*ocr.Init("C:/Program Files/tesseract/bin/tessdata/", "eng", OEM_TESSERACT_LSTM_COMBINED);
	ocr.SetPageSegMode(PSM_SINGLE_CHAR);
	ocr.SetImage(cutImg[2].data, cutImg[2].cols, cutImg[2].rows, 1, cutImg[2].step);
	const char* text = ocr.GetUTF8Text();
	cout << "Text:" << endl;
	cout << text << endl;
	cout << "Confidence:" << ocr.MeanTextConf() << endl;*/
	
	waitKey(0);
}

Mat binarize(Mat input) {
	Mat gaussianImg;
	GaussianBlur(input, gaussianImg, Size(5, 5), 0,0);
	//imshow("gaussian blur", gaussianImg);

	Mat hsvImg;
	cvtColor(gaussianImg, hsvImg, COLOR_BGR2HSV);
	Mat mask;
	inRange(hsvImg, Scalar(100, 43, 46), Scalar(124, 255, 255), mask);

	vector<Mat>hsv;
	split(hsvImg, hsv);
	Mat sImg = hsv[1];
	Mat grayImg = Mat::zeros(input.size(), CV_8U);
	for (int row = 0; row < sImg.rows; row++) {
		for (int col = 0; col < sImg.cols; col++) {
			if (mask.at<uchar>(row, col) > 0) {
				grayImg.at<uchar>(row, col) = sImg.at<uchar>(row, col);
			}
		}
	}
	//imshow("gray", grayImg);

	Mat blurImg;
	GaussianBlur(grayImg, blurImg, Size(5, 5), 0, 0);
	//imshow("blur image", blurImg);

	Mat sobelImg;
	Sobel(blurImg, sobelImg, CV_16S, 1, 0, 3);
	convertScaleAbs(sobelImg, sobelImg);
	sobelImg.convertTo(sobelImg, CV_8U);
	//imshow("sobel image", sobelImg);

	Mat blurAgain;
	medianBlur(sobelImg, blurAgain, 3);
	//imshow("blur2 image", blurAgain);

	Mat binaryImg;
	threshold(blurAgain, binaryImg, 110, 250, THRESH_BINARY);
	//imshow("binary image", binaryImg);

	Mat closeImg;
	Mat kl = getStructuringElement(MORPH_RECT, Size(5, 3));
	morphologyEx(binaryImg, closeImg, MORPH_CLOSE, kl, Point(-1, -1), closeTimes);
	//imshow("close image", closeImg);

	return closeImg;
}

RotatedRect GetLocatedArea(Mat input){
	vector<vector<Point>>contours;
	vector<Vec4i>hierachy;
	findContours(input, contours, hierachy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));

	vector<vector<Point>>contours_poly(contours.size());
	Rect testRect;
	vector<RotatedRect>MinRect(contours.size());
	RotatedRect locatedRect(Point2f(10, 10), Size2f(10, 10), 0);
	for (size_t i = 0; i < contours.size(); i++) {
		approxPolyDP(contours[i], contours_poly[i], 3, true);
		testRect = boundingRect(contours_poly[i]);
		int whiteArea = 0;
		for (int row = testRect.tl().y; row < testRect.br().y; row++) {
			for (int col = testRect.tl().x; col < testRect.br().x; col++) {
				if (input.at<uchar>(row, col)) {
				whiteArea++;
				}
			}
		}
	 //if (static_cast<float>(whiteArea) / (locatedRect.size.width * locatedRect.size.height) < 0.7) {
			//continue;
		//}
		MinRect[i] = minAreaRect(contours_poly[i]);
		float width = MinRect[i].size.width;
		float height = MinRect[i].size.height;
		float index = width / height;
		float minVal = 2.8;
		float maxVal = 4.6;
		if (MinRect[i].size.width <  MinRect[i].size.height) {
			minVal = 0.22;
			maxVal = 0.36;
		}
		if (index > minVal && index < maxVal && (width * height) >(locatedRect.size.width * locatedRect.size.height)) {
			locatedRect = MinRect[i];
		}
	}

	return locatedRect;
}

Mat getNumbers(RotatedRect box) {
	Mat gray;
	cvtColor(src, gray, COLOR_BGR2GRAY);

	Mat rotated;
	double angle = box.angle;
	auto size = box.size;
	if (box.angle - 45.0 && size.width < size.height) {
		angle += 90.0;
		swap(size.width, size.height);
	}
	auto transform = getRotationMatrix2D(box.center, angle, 1);
	warpAffine(gray, rotated, transform, gray.size(), INTER_CUBIC);
	Mat cropped;
	getRectSubPix(rotated, size, box.center, cropped);

	Mat number;
	resize(cropped, number, Size(490, 150), 0, 0);
	medianBlur(number, number, 3);
	Mat kernel = (Mat_<int>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
	filter2D(number, number, -1, kernel);
	threshold(number, number, 150, 255, THRESH_BINARY | THRESH_OTSU);

	return number;
}

vector<Mat>cutNumbers(Mat input) {
	int cutWidth[8];
	cutWidth[0] = 0;
	cutWidth[7] = 489;
	for (int index = 1; index <= 6; index++) {
		int min = 150;
		for (int col = 70 * index - 20; col < 70 * index + 20; col++) {
			int whiteArea = 0;
			for (int row = 0; row < 150; row++) {
				if (input.at<uchar>(row, col)) {
					whiteArea++;
				}
			}
			if (whiteArea <= min) {
				min = whiteArea;
				cutWidth[index] = col;
			}
		}
	}

	vector<Mat>cutNumbers(7);
	for (int n = 0; n < 7; n++) {
		cutNumbers[n] = Mat::zeros(150, cutWidth[n + 1] - cutWidth[n], CV_8U);
		int a = 0, b = 0;
		for (int y = 0; y < 150; y++) {
			b = 0;
			for (int x = cutWidth[n]; x < cutWidth[n + 1]; x++) {
				cutNumbers[n].at<uchar>(a, b) = input.at<uchar>(y, x);
				b++;
			}
			a++;
		}
		resize(cutNumbers[n], cutNumbers[n], Size(70, 150), 0, 0);
	}

	vector<Mat>minNumbers(7);
	for (int i = 0; i < 7; i++) {
		vector<vector<Point>>number_contour;
		findContours(cutNumbers[i], number_contour, RETR_TREE, CHAIN_APPROX_SIMPLE);
		Rect number_rect;
		Rect finalRect(0, 0, 1, 1);
		float whiteTemp = 0;
		for (int k = 0; k < number_contour.size(); k++) {
			int whiteArea = 0;
			number_rect = boundingRect(number_contour[k]);
			for (int row = number_rect.tl().y; row < number_rect.br().y; row++) {
				for (int col = number_rect.tl().x; col < number_rect.br().x; col++) {
					if (cutNumbers[i].at<uchar>(row, col) == 255) {
						whiteArea++;
					}
				}
			}
			if ( number_rect.area()>3200 && static_cast<float>(whiteArea) / static_cast<float>(number_rect.area()) > whiteTemp 
				&& static_cast<float>(number_rect.size().height) / static_cast<float>(number_rect.size().width) > 1.5) {
				whiteTemp = static_cast<float>(whiteArea) / static_cast<float>(number_rect.area());
				finalRect = number_rect;
			}
			if (number_rect.area() > 1000 && number_rect.area() < 3000 && static_cast<float>(whiteArea) / static_cast<float>(number_rect.area()) > 0.5 
				&& static_cast<float>(number_rect.size().height) / static_cast<float>(number_rect.size().width) > 2.5) {
				finalRect = number_rect;
				break;
			}
		}
		
		getRectSubPix(cutNumbers[i], finalRect.size(), Point((finalRect.tl().x + finalRect.br().x) / 2, (finalRect.tl().y + finalRect.br().y) / 2), minNumbers[i]);
		//cout << finalRect.area() << '\n' << endl;
		resize(minNumbers[i], minNumbers[i], Size(60, 150), 0, 0);
		//imshow("2", minNumbers[i]);
		//waitKey(0);
	}
		return minNumbers;
}

string numberIdentify(vector<Mat>input) {
	vector<Mat>models(64);
	vector<string>filename{ "F:/ps/chepai/0.jpg","F:/ps/chepai/1.jpg" ,"F:/ps/chepai/2.jpg" ,"F:/ps/chepai/3.jpg" ,"F:/ps/chepai/4.jpg" ,"F:/ps/chepai/5.jpg" ,
		"F:/ps/chepai/6.jpg" ,"F:/ps/chepai/7.jpg" ,"F:/ps/chepai/8.jpg" ,"F:/ps/chepai/9.jpg" ,"F:/ps/chepai/A.jpg" ,"F:/ps/chepai/B.jpg" ,
		"F:/ps/chepai/C.jpg" ,"F:/ps/chepai/D.jpg" ,"F:/ps/chepai/E.jpg" ,"F:/ps/chepai/F.jpg" ,"F:/ps/chepai/G.jpg" ,"F:/ps/chepai/H.jpg" ,
		"F:/ps/chepai/J.jpg" ,"F:/ps/chepai/K.jpg" ,"F:/ps/chepai/L.jpg" ,"F:/ps/chepai/M.jpg" ,"F:/ps/chepai/N.jpg" ,"F:/ps/chepai/P.jpg" ,
		"F:/ps/chepai/Q.jpg" ,"F:/ps/chepai/R.jpg" ,"F:/ps/chepai/S.jpg" ,"F:/ps/chepai/T.jpg" ,"F:/ps/chepai/U.jpg" ,
		"F:/ps/chepai/V.jpg" ,"F:/ps/chepai/W.jpg" ,"F:/ps/chepai/X.jpg", "F:/ps/chepai/Y.jpg" ,"F:/ps/chepai/Z.jpg","F:/ps/chepai/jing.jpg" ,"F:/ps/chepai/jin.jpg" ,"F:/ps/chepai/ji.jpg" 
	,"F:/ps/chepai/jin2.jpg" ,"F:/ps/chepai/meng.jpg" ,"F:/ps/chepai/hei.jpg" ,"F:/ps/chepai/ji2.jpg" ,"F:/ps/chepai/liao.jpg" ,"F:/ps/chepai/hu.jpg" ,"F:/ps/chepai/su.jpg" ,"F:/ps/chepai/zhe.jpg" 
	,"F:/ps/chepai/wan.jpg" ,"F:/ps/chepai/gan.jpg" ,"F:/ps/chepai/min.jpg" ,"F:/ps/chepai/lu.jpg" ,"F:/ps/chepai/yu.jpg" ,"F:/ps/chepai/ee.jpg" 
	,"F:/ps/chepai/xiang.jpg" ,"F:/ps/chepai/yue.jpg" ,"F:/ps/chepai/gui.jpg" ,"F:/ps/chepai/yun.jpg" ,"F:/ps/chepai/gui2.jpg" ,"F:/ps/chepai/chuan.jpg","F:/ps/chepai/yu2.jpg" 
	,"F:/ps/chepai/zang.jpg" ,"F:/ps/chepai/shan.jpg" ,"F:/ps/chepai/gan2.jpg" ,"F:/ps/chepai/ning.jpg" ,"F:/ps/chepai/qing.jpg" ,"F:/ps/chepai/xin.jpg" };
	for (int i = 0; i < 64; i++) {
		models[i] = imread(filename[i]);
		cvtColor(models[i], models[i], COLOR_BGR2GRAY);
		
	}
		
	string temp = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ京津冀晋蒙黑吉辽沪苏浙皖赣闽鲁豫鄂湘粤桂云贵川渝藏陕甘宁青新";
	string number;
	int defeatTimes = 0;

	double max = 0;
	double maxVal;
	int finalNumber = 0;
	for (int j = 34; j < 64; j++) {
		Mat result = Mat::zeros(input[1].rows - models[j].rows + 1, input[1].cols - models[j].cols + 1, CV_32F);
		matchTemplate(input[1], models[j], result, TM_CCOEFF_NORMED, Mat());
		minMaxLoc(result, 0, &maxVal, 0, 0);
		if (maxVal > max) {
			max = maxVal;
			finalNumber = j;
		}
	}
		if (max < 0.3) {
			defeatTimes++;
		}
		number += temp[finalNumber];

	for (int i = 1; i < 7; i++) {
		max = 0;
		maxVal;
		finalNumber = 0;
		for(int j = 0; j < 34; j++) {
			Mat result = Mat::zeros(input[i].rows - models[j].rows + 1, input[i].cols - models[j].cols + 1, CV_32F);
			matchTemplate(input[i], models[j], result, TM_CCOEFF_NORMED, Mat());
			minMaxLoc(result, 0, &maxVal, 0, 0);
			if (maxVal > max) {
				max = maxVal;
				finalNumber = j;
			}
		}
		if (max < 0.3) {
			defeatTimes++;
		}
		number += temp[finalNumber];
	}

	if (defeatTimes > 4) {
		return "match defeat";
	}
	return number;
}
