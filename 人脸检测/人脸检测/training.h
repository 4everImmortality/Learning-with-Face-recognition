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
	// �����ͷ���һ����һ�����ͼ�����: 
	Mat dst;
	switch (src.channels())
	{
	case 1:
		normalize(_src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
		/*src �������飻
		dst ������飬����Ĵ�С��ԭ����һ�£�
		alpha �����淶ֵ���߹淶��Χ�����������ޣ�
		beta ֻ�����淶��Χ���������ޣ����ֻ��NORM_MINMAX�������ã�
		norm_type ��һ��ѡ�����ѧ��ʽ���ͣ�
		dtype ��Ϊ��������ڴ�С���ͨ�������������룬��Ϊ�������ֻ����������粻ͬ����ͬ �ĵط���dtype������
		mark ���롣ѡ�����Ȥ����ѡ����ֻ�ܶԸ�������в�����*/
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
	while (getline(file, line))//��ȡһ���ı�
	{
		stringstream lines(line);//���ַ����ķָ�
		getline(lines, path, spearator);//����ͼƬ�ļ�·��,��';'��Ϊ�޶���
		getline(lines, classlabel);//����ͼƬ��ǩ��Ĭ���޶���
		if (!path.empty() || !classlabel.empty())
		{
			images.push_back(imread(path, 0));
			labels.push_back(atoi(classlabel.c_str()));// ascii to integer
		}
	}
}
void training()
{
	//csv·��
	string csv_path = "";
	//ͼƬ�ͱ�ǩ
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


	// ��������ݼ����Ƴ����һ��ͼƬ����Ϊ����ͼƬ  
	Mat testSample = images[images.size() - 1];
	int testLabel = labels[labels.size() - 1];
	images.pop_back();//ɾ�����һ����Ƭ������Ƭ��Ϊ����ͼƬ
	labels.pop_back();//ɾ������һ����Ƭ��labels


	//����һ��PCA��������������ʱ����Ϊmodel��������ɺ�
	//�������еĳ�Ա����train()����ɷ�������ѵ��
	Ptr<cv::face::BasicFaceRecognizer> model = cv::face::EigenFaceRecognizer::create();
	model->train(images, labels);
	model->save("MyFacePCAModel.xml");

	Ptr<cv::face::BasicFaceRecognizer> model1 = cv::face::FisherFaceRecognizer::create();
	model1->train(images, labels);
	model1->save("MyFaceFisherModel.xml");

	Ptr<cv::face::LBPHFaceRecognizer> model2 = cv::face::LBPHFaceRecognizer::create();
	model2->train(images, labels);
	model2->save("MyFaceLBHPModel.xml");

	// ����Բ���ͼ�����Ԥ�⣬predictedLabel��Ԥ���ǩ���  
	//ע��predict()��ڲ�������Ϊ��ͨ���Ҷ�ͼ�����ͼ�����Ͳ�������Ҫ�Ƚ���ת��
	//predict()��������һ�����α�����Ϊʶ���ǩ
	int predictedLabel = model->predict(testSample);//���ط�����
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