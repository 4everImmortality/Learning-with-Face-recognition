#pragma once
#include<iostream>
#include<opencv2/opencv.hpp>
#include<vector>

using namespace std;
using namespace cv;

void capturePeople()
{
	//加载分类器
	CascadeClassifier cascade;
	cascade.load("D:/opencv/sources/data/haarcascades/haarcascade_frontalface_alt2.xml");
	
	VideoCapture cap(0);
	Mat frame, myFace;
	int pic_num = 1;
	while (1)
	{
		cap >> frame;
		vector<Rect>faces;//vector存放检测到的人脸
		Mat frame_gray;
		cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
		//调用detectMultiScale()函数检测，调整函数的参数可以使检测结果更加精确。
		cascade.detectMultiScale(frame_gray, faces, 1.1, 4, 0, Size(30, 30), Size(1000, 1000));
		//参数1：表示的是要检测的输入图像 
		//参数2：表示检测到的人脸目标序列,
		//参数3：表示每次图像尺寸减小的比例
		//参数4：表示每一个目标至少要被检测到3次才算是真的目标(因为周围的像素和不同的窗口大小都可以检测到人脸表示每一个目标至少要被检测到3次才算是真的目标(因为周围的像素和不同的窗口大小都可以检测到人脸
		/*参数5：flags–要么使用默认值，要么使用CV_HAAR_DO_CANNY_PRUNING，
			函数将会使用Canny边缘检测来排除边缘过多或过少的区域，
			因为这些区域通常不会是人脸所在区域；opencv3 以后都不用这个参数了*/
			//参数6：Size(100, 100)为目标的最小尺寸 一般为30*30 是最小的了 也够了
			//参数7：Size(500, 500)为目标的最大尺寸 其实可以不用这个，opencv会自动去找这个最大尺寸
		cout << "the number of the faces:" << faces.size() << endl;
		//识别到的脸用矩形圈出
		for (int i = 0; i < faces.size(); i++)
		{
			rectangle(frame, faces[i], Scalar(0, 0, 255), 2, 8, 0);
		}
		//当只有一张人脸的时候，开始保存数据：
		if (faces.size() == 1)
		{
			Mat faceROI = frame_gray(faces[0]);//在灰度图中裁剪出图片
			resize(faceROI, myFace, Size(150, 150));
			putText(frame, to_string(pic_num), faces[0].tl(), 3, 1.2, Scalar(255, 0, 0), 2, 0);
			//在 faces[0]的topleft坐标上写序号
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
		imshow("frame", frame);//显示视频流
		waitKey(1000);
	}
}