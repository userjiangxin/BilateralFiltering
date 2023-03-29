#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace std;
using namespace cv;

void myBilateralFilter(const Mat &src, Mat &dst, int ksize, double space_sigma, double color_sigma)
{
	int channels = src.channels();//channels为src的通道数
	CV_Assert(channels == 1 || channels == 3);
	double space_coeff = -0.5 / (space_sigma * space_sigma);//space系数
	double color_coeff = -0.5 / (color_sigma * color_sigma);//color系数
	int radius = (ksize-1) / 2;
	Mat temp;

	copyMakeBorder(src, temp, radius, radius, radius, radius, cv::BorderTypes::BORDER_REFLECT);// 最上、最下各加1行，最左、最右各加radius列，增加的部分与 src 关于边界对称，temp为补齐后的图像。
	vector<double> _color_weight(channels * 256);// 由“像素值之差”决定的权重（模板上某像素与模板中心像素）。若 channels =1，“像素值之差”的值域为 [0,255] 。若 channels =3，“像素值之差”的值域为 [0,255*3] 。
	vector<double> _space_weight(ksize * ksize);// 由“像素之间的距离”决定的权重，是一维向量，大小为 ksize * ksize
	vector<int> _space_ofs(ksize * ksize);	// 模板上某像素相对模板中心像素的位置偏移量
	double *color_weight = &_color_weight[0];// color_weight 指向 _color_weight 的首元素
	double *space_weight = &_space_weight[0];	// space_weight 指向 _space_weight 的首元素
	int *space_ofs = &_space_ofs[0]; //space_ofs 指向 _space_ofs 的首元素
	for (int i = 0; i < channels * 256; i++)
	{
		color_weight[i] = exp(i * i * color_coeff);//这里生成一个固定的模板，描述像素值之差的权重。
	}

	int maxk = 0;
	for (int i = -radius; i <= radius; i++)
	{
		for (int j = -radius; j <= radius; j++) 
		{
			double r = sqrt(i * i + j * j);
			if (r > radius)
				continue;
			space_weight[maxk] = exp(r * r * space_coeff);
			//Mat.step为矩阵第一行元素的字节数
			space_ofs[maxk++] = i * temp.step + j * channels;//这里可以看图像在内存中存放的方式。
			// space_weight 与 space_ofs 均为一维向量，且长度相等，元素之间一一对应。某像素相对于中心像素的位置偏移量为 space_ofs[maxk]，该像素的空间权重为 space_weight[maxk]
		}
	}

	//滤波过程
	for (int i = 0; i < src.rows; i++)
	{
		// temp.data 指向 temp[0][0][B] ，temp.step 为 temp.cols * channels；i = 0 时，temp.data + (i + radius) * temp.step 指向 temp[radius][0][B]
		// i = 0 时，temp.data + (i + radius) * temp.step + radius * channels 指向 temp[radius][radius][B]，即 src[0][0][B] ，即 sptr 指向 src[0][0][B]
		// 一般的，sptr 指向 src[i][0][B]
		const uchar *sptr = temp.data + (i + radius)* temp.step + radius * channels;
		uchar *dptr = dst.data + i * dst.step;

		if (channels == 1)
		{
			for (int j = 0; j < src.cols; j++)
			{
				double sum = 0, wsum = 0;	// sum 为 temp_mask 中所有像素值的加权和，wsum 为 mask 中所有权重的和
				int val0 = sptr[j];// temp_mask 中心的像素值。j = 0 时，(sptr+j) 指向 src[i][0]。一般的，(sptr+j) 指向 src[i][j] ，即 temp_mask 的中心像素
				for (int k = 0; k < maxk; k++)
				{
					int val = sptr[j + space_ofs[k]];// 某像素 k 相对于中心像素的位置偏移量为 space_ofs[k]，该像素的空间权重为 space_weight[k]
					double w = space_weight[k] * color_weight[abs(val - val0)];// 模板系数 = 由像素距离决定的权重 * 由像素差值决定的权重
					sum += val * w;
					wsum += w;
				}
				dptr[j] = (uchar)cvRound(sum / wsum);
			}
		}
		else if (channels == 3)
		{
			for (int j = 0; j < src.cols * 3; j += 3)
			{

				double sum_b = 0, sum_g = 0, sum_r = 0, wsum = 0;
				int b0 = sptr[j];
				int g0 = sptr[j + 1];
				int r0 = sptr[j + 2];
				for (int k = 0; k < maxk; k++)
				{
					// j = 0 时，(sptr+j) 指向 src[i][0][B] ，即 temp_mask 的中心像素的 [B]；若 k 也为0，则 sptr + j + space_ofs[k] 指向中心像素左上角的那个像素的 [B]
					// 某像素 k 相对于中心像素的位置偏移量为 space_ofs[k]，该像素的空间权重为 space_weight[k]
					//const uchar *sptr_k = sptr + j + space_ofs[k];
					int b = sptr[j + space_ofs[k]];	// sptr_k+0 指向中心像素左上角的那个像素的 [B]
					int g = sptr[j + space_ofs[k] + 1];// sptr_k+1 指向中心像素左上角的那个像素的 [G]
					int r = sptr[j + space_ofs[k] + 2];	// sptr_k+2 指向中心像素左上角的那个像素的 [R]
					double w = space_weight[k] * color_weight[abs(b - b0) + abs(g - g0) + abs(r - r0)];
					sum_b += b * w;
					sum_g += g * w;
					sum_r += r * w;
					wsum += w;
				}
				wsum = 1.0f / wsum;
				dptr[j] = (uchar)cvRound(sum_b * wsum);// dptr 指向 dst[i][0][B] ，(dptr+j) 指向 dst[i][j/3][B]
				dptr[j + 1] = (uchar)cvRound(sum_g * wsum);// (dptr+j+1) 指向 dst[i][j/3][G]
				dptr[j + 2] = (uchar)cvRound(sum_r * wsum);// (dptr+j+2) 指向 dst[i][j/3][R]
				
			}
		}
	}


}
void myBilateralFilter1(const Mat &src, Mat &dst, int ksize, double space_sigma, double color_sigma)
{
	int channels = src.channels();
	CV_Assert(channels == 1 || channels == 3);
	double space_coeff = -0.5 / (space_sigma * space_sigma);
	double color_coeff = -0.5 / (color_sigma * color_sigma);
	int radius = (ksize -1) / 2;
	Mat temp;
	copyMakeBorder(src, temp, radius, radius, radius, radius, cv::BorderTypes::BORDER_REFLECT);
	vector<double> color_weight(channels * 256);
	Mat space_mask = cv::Mat::zeros(ksize, ksize, CV_64F);//创建个空域模板大小；
	for (int i = 0; i < channels * 256; i++)
	{
		color_weight[i] = exp(i * i * color_coeff);
	}
	for (int i = 0; i < space_mask.rows; i++)
	{
		double x = pow((i - radius), 2);
		//double * Maskdata = space_mask.ptr<double>(i);
		for (int j = 0; j < space_mask.cols; j++)
		{
			double y = pow((j - radius), 2);
			space_mask.at<double>(i, j) = exp((x + y) * space_coeff);//Maskdata[j] = ;

		}
	}
	//双边滤波
	if (channels == 1)
	{
		for (int i = radius; i < src.rows + radius; i++)
		{
			for (int j = radius; j < src.cols + radius; j++)
			{
				double sum = 0, wsum = 0;
				int val0 = temp.at<uchar>(i, j);
				for (int r = -radius; r <= radius; r++)
				{
					for (int c = -radius; c <= radius; c++)
					{
						int val = temp.at<uchar>(i + r, j + c);
						double w = space_mask.at<double>(r + radius, c + radius) * color_weight[abs(val - val0)];
						sum += val * w;
						wsum += w;
					}
				}
				dst.at<uchar>(i - radius, j - radius) = cvRound(sum / wsum);
			}

		}
	}

	else if (channels == 3)
	{
		for (int i = radius; i < src.rows + radius; i++)
		{
			for (int j = radius; j < src.cols + radius; j++)
			{
				double sum_b = 0, sum_g = 0, sum_r = 0, wsum = 0;
				Vec3b bgr0= temp.at<Vec3b>(i, j);
				
				for (int r = -radius; r <= radius; r++)
				{
					for (int c = -radius; c <= radius; c++)
					{
						Vec3b bgr = temp.at<Vec3b>(i + r, j + c);
						double w = space_mask.at<double>(r + radius, c + radius) * color_weight[abs(bgr[0] - bgr0[0]) + abs(bgr[1] - bgr0[1]) + abs(bgr[2] - bgr0[2])];
						sum_b += bgr[0] * w;
						sum_g += bgr[1] * w;
						sum_r += bgr[2] * w;
						wsum += w;
					}
				}
				Vec3b value = { (uchar)cvRound(sum_b / wsum) ,(uchar)cvRound(sum_g / wsum) ,(uchar)cvRound(sum_r / wsum) };
				dst.at<Vec3b>(i - radius, j - radius) =value;
				
			}
		}
	}
}

int main()
{
	string path = "D:\\opencv\\pictures\\lena.jpeg";
	//string path = "Resources/shanfeng.jpg";
	Mat src = imread(path, IMREAD_GRAYSCALE);
	Mat dst1 = src.clone();
	Mat dst2 = src.clone();
	int ksize = 20;
	double space_sigma = 80;
	double color_sigma = 80;
	myBilateralFilter(src, dst1, ksize, space_sigma, color_sigma);
	bilateralFilter(src, dst2, ksize, space_sigma, color_sigma);
	namedWindow("src", cv::WINDOW_AUTOSIZE);
	namedWindow("dst1-myBilateralFilter", cv::WINDOW_AUTOSIZE);
	namedWindow("dst2-opencv_bilateralFilter", cv::WINDOW_AUTOSIZE);
	imshow("src", src);
	imshow("dst1-myBilateralFilter", dst1);
	imshow("dst2-opencv_bilateralFilter", dst1);
	waitKey(0);
	return 0;
}