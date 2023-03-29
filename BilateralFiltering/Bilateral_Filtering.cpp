#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;


int g_d = 23;
int g_sigmaColor = 20;
int g_sigmaSpace = 50;

Mat image1;
Mat image2;
Mat image3;

void on_Trackbar(int, void*)
{
	bilateralFilter(image1, image2, g_d, g_sigmaColor, g_sigmaSpace);
	imshow("output", image2);
}

int main()
{
	image1 = imread("Resources/shanfeng.jpg");
	if (image1.empty())
	{
		cout << "Could not load image ... " << endl;
		return  -1;
	}

	image2 = Mat::zeros(image1.rows, image1.cols, image1.type());
	
	namedWindow("output");

	createTrackbar("核直径", "output", &g_d, 50, on_Trackbar);
	createTrackbar("颜色空间方差", "output", &g_sigmaColor, 100, on_Trackbar);
	createTrackbar("坐标空间方差", "output", &g_sigmaSpace, 100, on_Trackbar);
	//bilateralFilter(image1, image2, g_d, g_sigmaColor, g_sigmaSpace);
	on_Trackbar(0,NULL);
	GaussianBlur(image1, image3, Size(g_d, g_d), g_sigmaSpace);
	imshow("input", image1);
	imshow("output", image2);
	imshow("GaussianBlur", image3);

	waitKey(0);
	return 0;
}