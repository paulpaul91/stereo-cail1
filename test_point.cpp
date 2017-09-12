/******************************/
/*        立体匹配和测距        
     1.可直接得深度图，测距
     2.通过鼠标操作得每个位置的距离信息
*/
/******************************/

#include <opencv2/opencv.hpp>  
#include <iostream>  

using namespace std;
using namespace cv;

const int imageWidth = 640;                             //摄像头的分辨率  
const int imageHeight = 480;
Size imageSize = Size(imageWidth, imageHeight);

Mat rgbImageL, grayImageL;
Mat rgbImageR, grayImageR;
Mat rectifyImageL, rectifyImageR;

Rect validROIL;//图像校正之后，会对图像进行裁剪，这里的validROI就是指裁剪之后的区域  
Rect validROIR;

Mat mapLx, mapLy, mapRx, mapRy;     //映射表  
Mat Rl, Rr, Pl, Pr, Q;              //校正旋转矩阵R，投影矩阵P 重投影矩阵Q
Mat xyz;              //三维坐标

Point origin;         //鼠标按下的起始点
Rect selection;      //定义矩形选框
bool selectObject = false;    //是否选择对象

int blockSize = 0, uniquenessRatio =0, numDisparities=0;
//Ptr<StereoBM> bm = StereoBM::create(16, 9);
StereoBM bm;

/*
事先标定好的相机的参数
fx 0 cx
0 fy cy
0 0  1
*/
Mat cameraMatrixL = (Mat_<double>(3, 3) << 3.4997701592680050e+02, 0., 3.2466573641115451e+02, 0.,
       3.5023076509883663e+02, 1.8332137042480844e+02, 0., 0., 1.);
Mat distCoeffL = (Mat_<double>(5, 1) << -1.6508818380185400e-01, 1.6601761013853036e-02,
       5.2286999360442988e-04, 4.3658628368214520e-04,
       3.7618451397647057e-03 );

Mat cameraMatrixR = (Mat_<double>(3, 3) <<3.5041530704855978e+02, 0., 3.4205256749887792e+02, 0.,
       3.5061187294650313e+02, 2.0247996473732590e+02, 0., 0., 1. );
Mat distCoeffR = (Mat_<double>(5, 1) << -1.6517447847637071e-01, 1.3312512747101385e-02,
       3.4537702687064672e-04, 3.1746515393604133e-04,
       7.3160071215800518e-03);

Mat T = (Mat_<double>(3, 1) <<  -1.1969878244880168e+02, -2.1852808163380097e-01,
       4.6235440219385671e-01 );//T平移向量
//Mat rec = (Mat_<double>(3, 1) << -0.00306, -0.03207, 0.00206);//rec旋转向量
Mat R = (Mat_<double>(3, 3) <<  9.9986552184645128e-01, -1.7385738583006935e-03,
       1.6306918276084319e-02, 1.8079281486137651e-03,
       9.9998937988500036e-01, -4.2392821352159622e-03,
       -1.6299374789638693e-02, 4.2681937809505143e-03,
       9.9985804637624198e-01);//R 旋转矩阵


/*****立体匹配*****/
void stereo_match(int,void*)
{
    //bm->setBlockSize(2*blockSize+5);     //SAD窗口大小，5~21之间为宜
    //bm->setROI1(validROIL);
    //bm->setROI2(validROIR);
    //bm->setPreFilterCap(31);
   //bm->setMinDisparity(0);  //最小视差，默认值为0, 可以是负值，int型
    //bm->setNumDisparities(numDisparities*16+16);//视差窗口，即最大视差值与最小视差值之差,窗口大小必须是16的整数倍，int型
    //bm->setTextureThreshold(10); 
    //bm->setUniquenessRatio(uniquenessRatio);//uniquenessRatio主要可以防止误匹配
    //bm->setSpeckleWindowSize(100);
    //bm->setSpeckleRange(32);
    //bm->setDisp12MaxDiff(-1);
    Mat disp, disp8;
    //bm->compute(rectifyImageL, rectifyImageR, disp);//输入图像必须为灰度图
    bm.state->roi1 = validROIL;
    bm.state->roi2 = validROIR;
    bm.state->preFilterCap = 31;
    bm.state->SADWindowSize = 2*blockSize+5;                                     // ´°¿Ú´óÐ¡
    bm.state->minDisparity = 0;                                       // È·¶¨Æ¥ÅäËÑË÷´ÓÄÄÀï¿ªÊ¼  Ä¬ÈÏÖµÊÇ0
    bm.state->numberOfDisparities = numDisparities*16+16;                // ÔÚ¸ÃÊýÖµÈ·¶¨µÄÊÓ²î·¶Î§ÄÚ½øÐÐËÑË÷
    bm.state->textureThreshold = 10;//10                                  // ±£Ö¤ÓÐ×ã¹»µÄÎÆÀíÒÔ¿Ë·þÔëÉù
    bm.state->uniquenessRatio = uniquenessRatio;     //10                               // !!Ê¹ÓÃÆ¥Åä¹¦ÄÜÄ£Ê½
    bm.state->speckleWindowSize = 100;   //13                             // ¼ì²éÊÓ²îÁ¬Í¨ÇøÓò±ä»¯¶ÈµÄ´°¿Ú´óÐ¡, ÖµÎª 0 Ê±È¡Ïû speckle ¼ì²é
    bm.state->speckleRange = 32;    //32                                  // ÊÓ²î±ä»¯ãÐÖµ£¬µ±´°¿ÚÄÚÊÓ²î±ä»¯´óÓÚãÐÖµÊ±£¬¸Ã´°¿ÚÄÚµÄÊÓ²îÇåÁã£¬int ÐÍ
    bm.state->disp12MaxDiff = -1;
    bm(rectifyImageL, rectifyImageR, disp);
    disp.convertTo(disp8, CV_8U, 255 / ((numDisparities * 16 + 16)*16.));//计算出的视差是CV_16S格式
    reprojectImageTo3D(disp, xyz, Q, true); //在实际求距离时，ReprojectTo3D出来的X / W, Y / W, Z / W都要乘以16(也就是W除以16)，才能得到正确的三维坐标信息。
    xyz = xyz * 16;
    imshow("disparity", disp8);
}

/*****描述：鼠标操作回调*****/
static void onMouse(int event, int x, int y, int, void*)
{
    if (selectObject)
    {
        selection.x = MIN(x, origin.x);
        selection.y = MIN(y, origin.y);
        selection.width = std::abs(x - origin.x);
        selection.height = std::abs(y - origin.y);
    }

    switch (event)
    {
    case EVENT_LBUTTONDOWN:   //鼠标左按钮按下的事件
        origin = Point(x, y);
        selection = Rect(x, y, 0, 0);
        selectObject = true;
        cout << origin <<"in world coordinate is: " << xyz.at<Vec3f>(origin) << endl;
        break;
    case EVENT_LBUTTONUP:    //鼠标左按钮释放的事件
        selectObject = false;
        if (selection.width > 0 && selection.height > 0)
        break;
    }
}


/*****主函数*****/
int main()
{
    /*
    立体校正
    */
    //Rodrigues(rec, R); //Rodrigues变换
    stereoRectify(cameraMatrixL, distCoeffL, cameraMatrixR, distCoeffR, imageSize, R, T, Rl, Rr, Pl, Pr, Q, CALIB_ZERO_DISPARITY,
        0, imageSize, &validROIL, &validROIR);
    initUndistortRectifyMap(cameraMatrixL, distCoeffL, Rl, Pr, imageSize, CV_32FC1, mapLx, mapLy);
    initUndistortRectifyMap(cameraMatrixR, distCoeffR, Rr, Pr, imageSize, CV_32FC1, mapRx, mapRy);

    /*
    读取图片
    */
    rgbImageL = imread("data_test/left-0037.png", CV_LOAD_IMAGE_COLOR);
    cvtColor(rgbImageL, grayImageL, CV_BGR2GRAY);
    rgbImageR = imread("data_test/right-0037.png", CV_LOAD_IMAGE_COLOR);
    cvtColor(rgbImageR, grayImageR, CV_BGR2GRAY);

    imshow("ImageL Before Rectify", grayImageL);
    imshow("ImageR Before Rectify", grayImageR);

    /*
    经过remap之后，左右相机的图像已经共面并且行对准了
    */
    remap(grayImageL, rectifyImageL, mapLx, mapLy, INTER_LINEAR);
    remap(grayImageR, rectifyImageR, mapRx, mapRy, INTER_LINEAR);

    /*
    把校正结果显示出来
    */
    Mat rgbRectifyImageL, rgbRectifyImageR;
    cvtColor(rectifyImageL, rgbRectifyImageL, CV_GRAY2BGR);  //伪彩色图
    cvtColor(rectifyImageR, rgbRectifyImageR, CV_GRAY2BGR);

    //单独显示
    //rectangle(rgbRectifyImageL, validROIL, Scalar(0, 0, 255), 3, 8);
    //rectangle(rgbRectifyImageR, validROIR, Scalar(0, 0, 255), 3, 8);
    imshow("ImageL After Rectify", rgbRectifyImageL);
    imshow("ImageR After Rectify", rgbRectifyImageR);

    //显示在同一张图上
    Mat canvas;
    double sf;
    int w, h;
    sf = 600. / MAX(imageSize.width, imageSize.height);
    w = cvRound(imageSize.width * sf);
    h = cvRound(imageSize.height * sf);
    canvas.create(h, w * 2, CV_8UC3);   //注意通道

    //左图像画到画布上
    Mat canvasPart = canvas(Rect(w * 0, 0, w, h));                                //得到画布的一部分  
    resize(rgbRectifyImageL, canvasPart, canvasPart.size(), 0, 0, INTER_AREA);     //把图像缩放到跟canvasPart一样大小  
    Rect vroiL(cvRound(validROIL.x*sf), cvRound(validROIL.y*sf),                //获得被截取的区域    
        cvRound(validROIL.width*sf), cvRound(validROIL.height*sf));
    //rectangle(canvasPart, vroiL, Scalar(0, 0, 255), 3, 8);                      //画上一个矩形  
    cout << "Painted ImageL" << endl;

    //右图像画到画布上
    canvasPart = canvas(Rect(w, 0, w, h));                                      //获得画布的另一部分  
    resize(rgbRectifyImageR, canvasPart, canvasPart.size(), 0, 0, INTER_LINEAR);
    Rect vroiR(cvRound(validROIR.x * sf), cvRound(validROIR.y*sf),
        cvRound(validROIR.width * sf), cvRound(validROIR.height * sf));
    //rectangle(canvasPart, vroiR, Scalar(0, 0, 255), 3, 8);
    cout << "Painted ImageR" << endl;

    //画上对应的线条
    for (int i = 0; i < canvas.rows; i += 16)
        line(canvas, Point(0, i), Point(canvas.cols, i), Scalar(0, 255, 0), 1, 8);
    imshow("rectified", canvas);

    /*
    立体匹配
    */
    namedWindow("disparity", CV_WINDOW_AUTOSIZE);
    // 创建SAD窗口 Trackbar
    createTrackbar("BlockSize:\n", "disparity",&blockSize, 8, stereo_match);
    // 创建视差唯一性百分比窗口 Trackbar
    createTrackbar("UniquenessRatio:\n", "disparity", &uniquenessRatio, 50, stereo_match);
    // 创建视差窗口 Trackbar
    createTrackbar("NumDisparities:\n", "disparity", &numDisparities, 16, stereo_match);
    //鼠标响应函数setMouseCallback(窗口名称, 鼠标回调函数, 传给回调函数的参数，一般取0)
    setMouseCallback("disparity", onMouse, 0);
    stereo_match(0,0);

    waitKey(0);
    return 0;
}
