#include<iostream>
#include<opencv2/opencv.hpp>
#include<vector>
#include"capturePeople.h"
#include"training.h"
using namespace std;
using namespace cv;
using namespace cv::face;

RNG g_rng(12345);
Ptr<FaceRecognizer> model;

int predict(Mat &src_image)//识别图片
{
	if (src_image.empty())
	{
		cout << "faile to read image !" << endl;
		return;
	}
	Mat face_test;
	int predict = 0;
	resize(src_image, face_test, Size(150, 150));
	if (face_test.channels() != 1)
	{
		face_test.reshape(1);
	}
	predict = model->predict(face_test);
	return predict;
}
int main(int agrc, char* argv[])
{
	//准备人脸数据集
	//capturePeople();//打开摄像头拍照
	//训练数据集
	//training();
	VideoCapture cap(0);
	if (!cap.isOpened())
	{
		cout << "failed to open camera !" << endl;
		return -1;
	}
	//加载分类器
	CascadeClassifier cascade;
	cascade.load("D:/opencv/sources/data/haarcascades/haarcascade_frontalface_alt2.xml");
	
	model = FisherFaceRecognizer::create();
	model->read("MyFaceFisherModel.xml");

	Mat frame, gray;
	while (true)
	{
		cap >> frame;
		vector<Rect>faces;
		cvtColor(frame, gray, COLOR_BGR2GRAY);
		equalizeHist(gray, gray);//均值化
		//检测人脸
		cascade.detectMultiScale(gray, faces, 1.1, 4, 0, Size(30, 30), Size(1000, 1000));
		
		Mat* pImage_roi = new Mat[faces.size()];    //ROI区域选出来
		Mat face;
		Point text_point;
		string str;
		//框脸:
		for (int i = 0; i < faces.size(); i++)
		{
			pImage_roi[i] = gray(faces[i]);
			text_point = Point(faces[i].x, faces[i].y);
			text_point = Point(faces[i].x, faces[i].y);
			if (pImage_roi->empty())
			{
				continue;
			}
			switch (predict(pImage_roi[i]))
			{
			case 41:str = "Man"; 
				break;
			case 42:str = "Women"; 
				break;
			case 43:str = "Boy"; 
				break;
			case 44:str = "Girl";
				break;
			case 45:str = "Baby";
				break;
			default: str = "Error"; 
				break;
			}
			Scalar color = Scalar(g_rng.uniform(0, 255), g_rng.uniform(0, 255), g_rng.uniform(0, 255));//所取的颜色任意值
			rectangle(frame, Point(faces[i].x, faces[i].y), Point(faces[i].x + faces[i].width, faces[i].y + faces[i].height), color, 1, 8);//放入缓存
			putText(frame, str, text_point, FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 255));//添加文字
		}
		delete[]pImage_roi;
		imshow("face", frame);
		waitKey(200);
	}
	return 0;
}