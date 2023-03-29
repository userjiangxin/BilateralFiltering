#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>

using namespace std;
using namespace cv;

void myBilateralFilter(const Mat &src, Mat &dst, int ksize, double space_sigma, double color_sigma)
{
	int channels = src.channels();//channelsΪsrc��ͨ����
	CV_Assert(channels == 1 || channels == 3);
	double space_coeff = -0.5 / (space_sigma * space_sigma);//spaceϵ��
	double color_coeff = -0.5 / (color_sigma * color_sigma);//colorϵ��
	int radius = (ksize-1) / 2;
	Mat temp;

	copyMakeBorder(src, temp, radius, radius, radius, radius, cv::BorderTypes::BORDER_REFLECT);// ���ϡ����¸���1�У��������Ҹ���radius�У����ӵĲ����� src ���ڱ߽�Գƣ�tempΪ������ͼ��
	vector<double> _color_weight(channels * 256);// �ɡ�����ֵ֮�������Ȩ�أ�ģ����ĳ������ģ���������أ����� channels =1��������ֵ֮���ֵ��Ϊ [0,255] ���� channels =3��������ֵ֮���ֵ��Ϊ [0,255*3] ��
	vector<double> _space_weight(ksize * ksize);// �ɡ�����֮��ľ��롱������Ȩ�أ���һά��������СΪ ksize * ksize
	vector<int> _space_ofs(ksize * ksize);	// ģ����ĳ�������ģ���������ص�λ��ƫ����
	double *color_weight = &_color_weight[0];// color_weight ָ�� _color_weight ����Ԫ��
	double *space_weight = &_space_weight[0];	// space_weight ָ�� _space_weight ����Ԫ��
	int *space_ofs = &_space_ofs[0]; //space_ofs ָ�� _space_ofs ����Ԫ��
	for (int i = 0; i < channels * 256; i++)
	{
		color_weight[i] = exp(i * i * color_coeff);//��������һ���̶���ģ�壬��������ֵ֮���Ȩ�ء�
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
			//Mat.stepΪ�����һ��Ԫ�ص��ֽ���
			space_ofs[maxk++] = i * temp.step + j * channels;//������Կ�ͼ�����ڴ��д�ŵķ�ʽ��
			// space_weight �� space_ofs ��Ϊһά�������ҳ�����ȣ�Ԫ��֮��һһ��Ӧ��ĳ����������������ص�λ��ƫ����Ϊ space_ofs[maxk]�������صĿռ�Ȩ��Ϊ space_weight[maxk]
		}
	}

	//�˲�����
	for (int i = 0; i < src.rows; i++)
	{
		// temp.data ָ�� temp[0][0][B] ��temp.step Ϊ temp.cols * channels��i = 0 ʱ��temp.data + (i + radius) * temp.step ָ�� temp[radius][0][B]
		// i = 0 ʱ��temp.data + (i + radius) * temp.step + radius * channels ָ�� temp[radius][radius][B]���� src[0][0][B] ���� sptr ָ�� src[0][0][B]
		// һ��ģ�sptr ָ�� src[i][0][B]
		const uchar *sptr = temp.data + (i + radius)* temp.step + radius * channels;
		uchar *dptr = dst.data + i * dst.step;

		if (channels == 1)
		{
			for (int j = 0; j < src.cols; j++)
			{
				double sum = 0, wsum = 0;	// sum Ϊ temp_mask ����������ֵ�ļ�Ȩ�ͣ�wsum Ϊ mask ������Ȩ�صĺ�
				int val0 = sptr[j];// temp_mask ���ĵ�����ֵ��j = 0 ʱ��(sptr+j) ָ�� src[i][0]��һ��ģ�(sptr+j) ָ�� src[i][j] ���� temp_mask ����������
				for (int k = 0; k < maxk; k++)
				{
					int val = sptr[j + space_ofs[k]];// ĳ���� k ������������ص�λ��ƫ����Ϊ space_ofs[k]�������صĿռ�Ȩ��Ϊ space_weight[k]
					double w = space_weight[k] * color_weight[abs(val - val0)];// ģ��ϵ�� = �����ؾ��������Ȩ�� * �����ز�ֵ������Ȩ��
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
					// j = 0 ʱ��(sptr+j) ָ�� src[i][0][B] ���� temp_mask ���������ص� [B]���� k ҲΪ0���� sptr + j + space_ofs[k] ָ�������������Ͻǵ��Ǹ����ص� [B]
					// ĳ���� k ������������ص�λ��ƫ����Ϊ space_ofs[k]�������صĿռ�Ȩ��Ϊ space_weight[k]
					//const uchar *sptr_k = sptr + j + space_ofs[k];
					int b = sptr[j + space_ofs[k]];	// sptr_k+0 ָ�������������Ͻǵ��Ǹ����ص� [B]
					int g = sptr[j + space_ofs[k] + 1];// sptr_k+1 ָ�������������Ͻǵ��Ǹ����ص� [G]
					int r = sptr[j + space_ofs[k] + 2];	// sptr_k+2 ָ�������������Ͻǵ��Ǹ����ص� [R]
					double w = space_weight[k] * color_weight[abs(b - b0) + abs(g - g0) + abs(r - r0)];
					sum_b += b * w;
					sum_g += g * w;
					sum_r += r * w;
					wsum += w;
				}
				wsum = 1.0f / wsum;
				dptr[j] = (uchar)cvRound(sum_b * wsum);// dptr ָ�� dst[i][0][B] ��(dptr+j) ָ�� dst[i][j/3][B]
				dptr[j + 1] = (uchar)cvRound(sum_g * wsum);// (dptr+j+1) ָ�� dst[i][j/3][G]
				dptr[j + 2] = (uchar)cvRound(sum_r * wsum);// (dptr+j+2) ָ�� dst[i][j/3][R]
				
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
	Mat space_mask = cv::Mat::zeros(ksize, ksize, CV_64F);//����������ģ���С��
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
	//˫���˲�
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