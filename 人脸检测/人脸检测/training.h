#pragma once
#include<opencv2/opencv.hpp>
#include<opencv2/face.hpp>
#include<opencv2/core.hpp>
#include<fstream>
#include<iostream>
#include<sstream>
#include<iostream>
#include<string>

using namespace std;
using namespace cv;

static Mat norm_0_255(InputArray _src) 
{
	Mat src = _src.getMat();
	// 创建和返回一个归一化后的图像矩阵: 
	Mat dst;
	switch (src.channels())
	{
	case 1:
		normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
		/*src 输入数组；
		dst 输出数组，数组的大小和原数组一致；
		alpha 用来规范值或者规范范围，并且是下限；
		beta 只用来规范范围并且是上限，因此只在NORM_MINMAX中起作用；
		norm_type 归一化选择的数学公式类型；
		dtype 当为负，输出在大小深度通道数都等于输入，当为正，输出只在深度与输如不同，不同 的地方由dtype决定；
		mark 掩码。选择感兴趣区域，选定后只能对该区域进行操作。*/
		break;
	case 3:
		normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
		break;
	default:
		src.copyTo(dst);
		break;
	}
	return dst;
}
static void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels, char spearator = ';')
{
	ifstream file (filename.c_str(), ifstream::in);
	if (!file)
	{
		cout << "failed to read csv file !" << endl;
		return;
	}
	string line, path, classlabel;
	while (getline(file, line))//读取一行文本
	{
		stringstream lines(line);//做字符串的分割
		getline(lines, path, spearator);//读入图片文件路径,以';'作为限定符
		getline(lines, classlabel);//读入图片标签，默认限定符
		if (!path.empty() || !classlabel.empty())
		{
			images.push_back(imread(path, 0));
			labels.push_back(atoi(classlabel.c_str()));// ascii to integer
		}
	}
}
void training()
{
	//csv路径
	string csv_path = "";
	//图片和标签
	vector<Mat>images;
	vector<int>labels;
	read_csv(csv_path, images, labels);
	for (int i = 0; i < images.size(); i++)
	{
		if (images.size() <= 1)
		{
			cout << "wrong csv file" << endl;
			return;
		}
		if (images[i].size() != Size(150, 150))
		{
			resize(images[i], images[i], Size(150, 150));
		}
	}
	cout << "sizes r ok !" << endl;


	// 从你的数据集中移除最后一张图片，作为测试图片  
	Mat testSample = images[images.size() - 1];
	int testLabel = labels[labels.size() - 1];
	images.pop_back();//删除最后一张照片，此照片作为测试图片
	labels.pop_back();//删除最有一张照片的labels


	//创建一个PCA人脸分类器，暂时命名为model，创建完成后
	//调用其中的成员函数train()来完成分类器的训练
	Ptr<cv::face::BasicFaceRecognizer> model = cv::face::EigenFaceRecognizer::create();
	model->train(images, labels);
	model->save("MyFacePCAModel.xml");

	Ptr<cv::face::BasicFaceRecognizer> model1 = cv::face::FisherFaceRecognizer::create();
	model1->train(images, labels);
	model1->save("MyFaceFisherModel.xml");

	Ptr<cv::face::LBPHFaceRecognizer> model2 = cv::face::LBPHFaceRecognizer::create();
	model2->train(images, labels);
	model2->save("MyFaceLBHPModel.xml");

	// 下面对测试图像进行预测，predictedLabel是预测标签结果  
	//注意predict()入口参数必须为单通道灰度图像，如果图像类型不符，需要先进行转换
	//predict()函数返回一个整形变量作为识别标签
	int predictedLabel = model->predict(testSample);//加载分类器
	int predictedLabel1 = model1->predict(testSample);
	int predictedLabel2 = model2->predict(testSample);

	string result_message = format("Predicted class = %d / Actual class = %d.", predictedLabel, testLabel);
	string result_message1 = format("Predicted class = %d / Actual class = %d.", predictedLabel1, testLabel);
	string result_message2 = format("Predicted class = %d / Actual class = %d.", predictedLabel2, testLabel);
	cout << result_message << endl;
	cout << result_message1 << endl;
	cout << result_message2 << endl;

	getchar();
}