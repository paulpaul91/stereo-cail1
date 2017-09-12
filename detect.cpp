#include<opencv2/opencv.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<iostream>
using namespace std;
cv::Point getcenterPoint(cv::Rect rect)
{
	cv::Point cpt;
	cpt.x=rect.x+cvRound(rect.width/2.0);
	cpt.y=rect.y+cvRound(rect.height/2.0);
	return cpt;

}
int main()
{
	cv::CascadeClassifier mFaceDetector;
	cv::CascadeClassifier mEyeDetector;
	cv::CascadeClassifier mMouthDetector;
	cv::CascadeClassifier mNoseDetector;
	mFaceDetector.load("haarcascade_frontalcatface.xml");
	mEyeDetector.load("haarcascade_mcs_eyepair_big.xml");
	mMouthDetector.load("haarcascade_mcs_nose.xml");
	mNoseDetector.load("haarcascade_mcs_mouth.xml");
	cv::Mat frame;
	cv::Mat mElabImage;
    cv::VideoCapture cap(0);
	if(!cap.isOpened())
	{
	        cout<<"can't open camera";
		return -1;
	}
	while(true)
	{
		cap>>frame;
		frame.copyTo(mElabImage);
		float scaleFactor = 3.0f;
		vector< cv::Rect > faceVec;
		mFaceDetector.detectMultiScale(frame, faceVec, scaleFactor);

		int i, j;
		for (i = 0; i<faceVec.size(); i++)
		{
			cv::rectangle(mElabImage, faceVec[i], CV_RGB(255, 0, 0), 2);
			cv::Mat face = frame(faceVec[i]);
			cv:: Point point_center=getcenterPoint(faceVec[i]);
			cout<<"area"<<faceVec[i].area()<<endl;
			if(faceVec[i].area()<300)
				continue;
			cout<<"point_center"<<point_center<<endl;
		}
		cv::imshow("Extracted Frame", mElabImage);
		cv::waitKey(20);
	}

	return 1;

}
