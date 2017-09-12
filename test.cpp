#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/legacy/legacy.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <list>
#include <algorithm>
#include <iterator>
#include <cstdio>
#include <string>

using namespace cv;
using namespace std;

typedef unsigned int uint;

Size imgSize(672, 376);
Size patSize(11, 8);           
const double patLen = 5.0f;    
double imgScale = 1.0;          


vector<string> fileList;
void initFileList(string dir, int first, int last){
	fileList.clear();
	for(int cur = first; cur <= last; cur++){
		string str_file = dir + "/" + to_string(cur) + ".png";
		fileList.push_back(str_file);
	}
}



static void saveXYZ(string filename, const Mat& mat)
{
    const double max_z = 1.0e4;
    ofstream fp(filename);
	if (!fp.is_open())
    {  
         std::cout<<"dian yun da kai shi bai"<<endl;    
         fp.close();  
		 return ;
    }  

    for(int y = 0; y < mat.rows; y++)
    {
        for(int x = 0; x < mat.cols; x++)
        {
            Vec3f point = mat.at<Vec3f>(y, x);   
            if(fabs(point[2] - max_z) < FLT_EPSILON || fabs(point[2]) > max_z)   
				continue;   
            fp<<point[0]<<" "<<point[1]<<" "<<point[2]<<endl;
        }
    }
	fp.close();
}


//cun shu sicha shuju
void saveDisp(const string filename, const Mat& mat)		
{
	ofstream fp(filename, ios::out);
	fp<<mat.rows<<endl;
	fp<<mat.cols<<endl;
	for(int y = 0; y < mat.rows; y++)
	{
		for(int x = 0; x < mat.cols; x++)
		{
			double disp = mat.at<short>(y, x); 
			fp<<disp<<endl;       
		}
	}
	fp.close();
}


void F_Gray2Color(Mat gray_mat, Mat& color_mat)
{
	color_mat = Mat::zeros(gray_mat.size(), CV_8UC3);
	int rows = color_mat.rows, cols = color_mat.cols;
	
	Mat red = Mat(gray_mat.rows, gray_mat.cols, CV_8U);
	Mat green = Mat(gray_mat.rows, gray_mat.cols, CV_8U);
	Mat blue = Mat(gray_mat.rows, gray_mat.cols, CV_8U);
	Mat mask = Mat(gray_mat.rows, gray_mat.cols, CV_8U);

	subtract(gray_mat, Scalar(255), blue);         // blue(I) = 255 - gray(I)
	red = gray_mat.clone();                        // red(I) = gray(I)
	green = gray_mat.clone();                      // green(I) = gray(I),if gray(I) < 128

	compare(green, 128, mask, CMP_GE);             // green(I) = 255 - gray(I), if gray(I) >= 128
	subtract(green, Scalar(255), green, mask);
	convertScaleAbs(green, green, 2.0, 2.0);

	vector<Mat> vec;
	vec.push_back(red);
	vec.push_back(green);
	vec.push_back(blue);
	cv::merge(vec, color_mat);
}

Mat F_mergeImg(Mat img1, Mat disp8){
	Mat color_mat = Mat::zeros(img1.size(), CV_8UC3);

	Mat red = img1.clone();
	Mat green = disp8.clone();
	Mat blue = Mat::zeros(img1.size(), CV_8UC1);

	vector<Mat> vec;
	vec.push_back(red);
	vec.push_back(blue);
	vec.push_back(green);
	cv::merge(vec, color_mat);
	
	return color_mat;
}
//获取深度图
int stereoMatch(int picNum, 
				string intrinsic_filename="intrinsics.yml", 
				string extrinsic_filename="extrinsics.yml", 
				bool no_display=false, 
				string point_cloud_filename="output/point3D.txt"
				)
{

	int color_mode = 0;
	Mat rawImg = imread(fileList[picNum], color_mode);    
	//CvtColor(rawImg, rawImg, CV_BGR2GRAY);
	if(rawImg.empty()){
		std::cout<<"In Function stereoMatch, the Image is empty..."<<endl;
		return 0;
	}
	cout<<"test get image"<<endl;

	Rect leftRect(0, 0, imgSize.width, imgSize.height);
	Rect rightRect(imgSize.width, 0, imgSize.width, imgSize.height);
	Mat img1 = rawImg(leftRect);       
	Mat img2 = rawImg(rightRect);      
    if(imgScale != 1.f){
        Mat temp1, temp2;
        int method = imgScale < 1 ? INTER_AREA : INTER_CUBIC;
        resize(img1, temp1, Size(), imgScale, imgScale, method);
        img1 = temp1;
        resize(img2, temp2, Size(), imgScale, imgScale, method);
        img2 = temp2;
    }
	imwrite("output/origionleft.jpg", img1);
	imwrite("output/origionright.jpg", img2);

    Size img_size = img1.size();

    // reading intrinsic parameters
	FileStorage fs(intrinsic_filename, CV_STORAGE_READ);
    if(!fs.isOpened())
    {
        std::cout<<"Failed to open file "<<intrinsic_filename<<endl;
        return -1;
    }
	Mat M1, D1, M2, D2;      
    fs["cameraMatrixL"] >> M1;
    fs["cameraDistcoeffL"] >> D1;
	fs["cameraMatrixR"] >> M2;
	fs["cameraDistcoeffR"] >> D2;

    M1 *= imgScale;
    M2 *= imgScale;

    fs.open(extrinsic_filename, CV_STORAGE_READ);
    if(!fs.isOpened())
    {
        std::cout<<"Failed to open file  "<<extrinsic_filename<<endl;
        return -1;
    }


	Rect roi1, roi2;
    Mat Q;
    Mat R, T, R1, P1, R2, P2;
    fs["R"] >> R;
    fs["T"] >> T;

    cv::stereoRectify( M1, D1, M2, D2, img_size, R, T, R1, R2, P1, P2, Q, CALIB_ZERO_DISPARITY, -1, img_size, &roi1, &roi2 );

	
    Mat map11, map12, map21, map22;
	initUndistortRectifyMap(M1, D1, R1, P1, img_size, CV_16SC2, map11, map12);
    initUndistortRectifyMap(M2, D2, R2, P2, img_size, CV_16SC2, map21, map22);

	
    Mat img1r, img2r;
    remap(img1, img1r, map11, map12, INTER_LINEAR);
    remap(img2, img2r, map21, map22, INTER_LINEAR);
    img1 = img1r;
    img2 = img2r;

	
	StereoBM bm;
	
	int unitDisparity = 15;//40
	int numberOfDisparities = unitDisparity * 16;
	bm.state->roi1 = roi1;
    bm.state->roi2 = roi2;
    bm.state->preFilterCap = 13;
    bm.state->SADWindowSize = 19;                                     // ´°¿Ú´óÐ¡
    bm.state->minDisparity = 0;                                       // È·¶¨Æ¥ÅäËÑË÷´ÓÄÄÀï¿ªÊ¼  Ä¬ÈÏÖµÊÇ0
    bm.state->numberOfDisparities = numberOfDisparities;                // ÔÚ¸ÃÊýÖµÈ·¶¨µÄÊÓ²î·¶Î§ÄÚ½øÐÐËÑË÷
    bm.state->textureThreshold = 1000;//10                                  // ±£Ö¤ÓÐ×ã¹»µÄÎÆÀíÒÔ¿Ë·þÔëÉù
    bm.state->uniquenessRatio = 1;     //10                               // !!Ê¹ÓÃÆ¥Åä¹¦ÄÜÄ£Ê½
    bm.state->speckleWindowSize = 200;   //13                             // ¼ì²éÊÓ²îÁ¬Í¨ÇøÓò±ä»¯¶ÈµÄ´°¿Ú´óÐ¡, ÖµÎª 0 Ê±È¡Ïû speckle ¼ì²é
    bm.state->speckleRange = 32;    //32                                  // ÊÓ²î±ä»¯ãÐÖµ£¬µ±´°¿ÚÄÚÊÓ²î±ä»¯´óÓÚãÐÖµÊ±£¬¸Ã´°¿ÚÄÚµÄÊÓ²îÇåÁã£¬int 
    bm.state->disp12MaxDiff = -1;

	Mat disp, disp8;
    int64 t = getTickCount();
    bm(img1, img2, disp);
    t = getTickCount() - t;
    printf("li ti pi pei haoshi: %fms\n", t*1000/getTickFrequency());
	

    disp.convertTo(disp8, CV_8U, 255/(numberOfDisparities*16.));
	
	Mat vdispRGB = disp8;
	F_Gray2Color(disp8, vdispRGB);

	Mat merge_mat = F_mergeImg(img1, disp8);

	saveDisp("output/shichashuju.txt", disp);


    if(!no_display){
        imshow("left jiaozheng", img1);
		imwrite("output/left_undistortRectify.jpg", img1);
        imshow("right jiaozheng", img2);
		imwrite("output/right_undistortRectify.jpg", img2);
        imshow("shi cha tu", disp8);
		imwrite("output/shicha.jpg", disp8);
		imshow("shichatu caise", vdispRGB);
		imwrite("output/shicatu_caise.jpg", vdispRGB);
		imshow("jiaozheng_yushicha_hebing", merge_mat);
		imwrite("output/jiaozheng_yushicha_hebing.jpg", merge_mat);
        //cv::waitKey();
        std::cout<<endl;
    }
	cv::destroyAllWindows();


	cout<<endl<<"jisuan dianyun... "<<endl;
    Mat xyz;
    reprojectImageTo3D(disp, xyz, Q, true);    
	cv::destroyAllWindows(); 
	cout<<endl<<"save point point_cloud_filename... "<<endl;
	saveXYZ(point_cloud_filename, xyz);
    
    cout<<endl<<endl<<"jiesu"<<endl<<"Press any key to end... ";
	
	getchar(); 
	return 0;
}




int main(){
	string intrinsic_filename = "intrinsics.yml";
	string extrinsic_filename = "extrinsics.yml";
	string point_cloud_filename = "output/point3D.txt";

	initFileList("data_test", 1, 1);
	stereoMatch(0, intrinsic_filename, extrinsic_filename, false, point_cloud_filename);

	return 0;
}



