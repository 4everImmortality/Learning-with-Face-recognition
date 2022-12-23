#pragma once
#include<iostream>
#include<opencv2/opencv.hpp>
#include<vector>

using namespace std;
using namespace cv;

void capturePeople()
{
	//���ط�����
	CascadeClassifier cascade;
	cascade.load("D:/opencv/sources/data/haarcascades/haarcascade_frontalface_alt2.xml");
	
	VideoCapture cap(0);
	Mat frame, myFace;
	int pic_num = 1;
	while (1)
	{
		cap >> frame;
		vector<Rect>faces;//vector��ż�⵽������
		Mat frame_gray;
		cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
		//����detectMultiScale()������⣬���������Ĳ�������ʹ��������Ӿ�ȷ��
		cascade.detectMultiScale(frame_gray, faces, 1.1, 4, 0, Size(30, 30), Size(1000, 1000));
		//����1����ʾ����Ҫ��������ͼ�� 
		//����2����ʾ��⵽������Ŀ������,
		//����3����ʾÿ��ͼ��ߴ��С�ı���
		//����4����ʾÿһ��Ŀ������Ҫ����⵽3�β��������Ŀ��(��Ϊ��Χ�����غͲ�ͬ�Ĵ��ڴ�С�����Լ�⵽������ʾÿһ��Ŀ������Ҫ����⵽3�β��������Ŀ��(��Ϊ��Χ�����غͲ�ͬ�Ĵ��ڴ�С�����Լ�⵽����
		/*����5��flags�CҪôʹ��Ĭ��ֵ��Ҫôʹ��CV_HAAR_DO_CANNY_PRUNING��
			��������ʹ��Canny��Ե������ų���Ե�������ٵ�����
			��Ϊ��Щ����ͨ��������������������opencv3 �Ժ󶼲������������*/
			//����6��Size(100, 100)ΪĿ�����С�ߴ� һ��Ϊ30*30 ����С���� Ҳ����
			//����7��Size(500, 500)ΪĿ������ߴ� ��ʵ���Բ��������opencv���Զ�ȥ��������ߴ�
		cout << "the number of the faces:" << faces.size() << endl;
		//ʶ�𵽵����þ���Ȧ��
		for (int i = 0; i < faces.size(); i++)
		{
			rectangle(frame, faces[i], Scalar(0, 0, 255), 2, 8, 0);
		}
		//��ֻ��һ��������ʱ�򣬿�ʼ�������ݣ�
		if (faces.size() == 1)
		{
			Mat faceROI = frame_gray(faces[0]);//�ڻҶ�ͼ�вü���ͼƬ
			resize(faceROI, myFace, Size(150, 150));
			putText(frame, to_string(pic_num), faces[0].tl(), 3, 1.2, Scalar(255, 0, 0), 2, 0);
			//�� faces[0]��topleft������д���
			string filename = format("%d.jpg", pic_num);
			imwrite(filename, myFace);
			imshow(filename, myFace);
			waitKey(1000);
			destroyAllWindows();
			pic_num++;
			if (pic_num > 10)
				break;
		}
		int c = waitKey(10);
		if ((char)c == 27) 
		{
			break;
		}
		imshow("frame", frame);//��ʾ��Ƶ��
		waitKey(1000);
	}
}