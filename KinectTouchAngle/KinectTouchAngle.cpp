// KinectTouchAngle.cpp : définit le point d'entrée pour l'application console.
//
#include "stdafx.h"
#include <iostream>
#include "OpenNI.h"
#include <vector>
#include <stack>

#include "opencv2/opencv.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace std;
using namespace openni;
using namespace cv;

int minY=56;
int maxY=200;
int minCr=12;
int maxCr=255;
int minCb=17;
int maxCb=255;

int dilationSize=1;
int erosionSize=2;

int positiveValue=200;
int negativeValue=200;

vector<vector<Point>>fingerSlices;

Mat derivX;
Mat derivY;
Mat deriv;

Mat derivXNormal;
Mat derivYNormal;
Mat derivNormal;


vector<Point> tableCorners(4);
RNG rng(12345);

enum ParentDirections{Y0=1,Y2=2,Non=5};
enum ExtendTypes{LeftRequired=1,RightRequired=2,AllRez=3,UnRez=4};


struct Span
{
	int XLeft;
	int XRight;
	int Y;
	ExtendTypes Extended;
	ParentDirections ParentDirection;
};
//vector<Span> spanContainer;
std::stack<Span> spanContainer;

void Process(Mat image,Point seed)
{
	uchar* imageP=(uchar*)image.data;
	*(imageP+seed.y*640+seed.x)=255;
}

bool IncludePredicate(Mat image,int x,int y)
{
	uchar* imageP=(uchar*)image.data;
	uchar aaa=*(imageP+y*640+x);
	if(*(imageP+y*640+x)==255)
		return true;
	else
		return false;
}
void CheckRange(Mat image,Mat& fillImage,int xleft,int xright,int y,ParentDirections ptype,Mat& maskImage)
{
	uchar* maskP=(uchar*)maskImage.data;

	for(int i=xleft;i<=xright;)
	{
		if(*(maskP+y*640+i)==0&&IncludePredicate(image,i,y))
		{
			int lb=i;
			int rb=i+1;
			while(rb<=xright&&*(maskP+y*640+rb)==0&&IncludePredicate(image,rb,y))
			{
				rb++;
			}
			rb--;

			Span span;
			span.XLeft=lb;
			span.XRight=rb;
			span.Y=y;

			if(lb==xleft&&rb==xright)
			{
				span.Extended=ExtendTypes::UnRez;
			}
			else if(rb==xright)
			{
				span.Extended=ExtendTypes::RightRequired;
			}
			else if(lb==xleft)
			{
				span.Extended=ExtendTypes::LeftRequired;
			}
			else
			{
				span.Extended=ExtendTypes::AllRez;
			}

			span.ParentDirection=ptype;

			for(int j=lb;j<=rb;++j)
			{
				*(maskP+y*640+j)=1;
				Process(fillImage,Point(j,y));
			}
			spanContainer.push(span);

			i=rb+1;
		}

		else
		{
			++i;
		}
	}
}

int FindXRight(Mat image,Mat& fillImage,Mat maskImage,int x, int y)
{
	int xright = x + 1;
	uchar* maskP=(uchar*)maskImage.data;

	while (true)
	{
		if (xright == image.cols ||*(maskP+y*640+xright)==1 )
		{
			break;
		}
		else
		{
			if (IncludePredicate(image,xright,y))
			{
				Point point(xright,y);	
				*(maskP+y*640+xright)=true;
				Process(fillImage,point);
				xright++;
			}
			else
			{
				break;
			}
		}
	}
	return xright - 1;
}

int FindXLeft(Mat image,Mat& fillImage,Mat maskImage,int x, int y)
{
	int xleft = x - 1;
	uchar* maskP=(uchar*)maskImage.data;
	while (true)
	{
		if (xleft == -1 || *(maskP+y*640+xleft)==1)
		{
			break;
		}
		else
		{
			if (IncludePredicate(image,xleft,y))
			{
				Point point(xleft,y);	
				*(maskP+y*640+xleft)=true;
				Process(fillImage,point);
				xleft--;
			}
			else
			{
				break;
			}
		}
	}
	return xleft + 1;
}

void ExcuteSpanFill(Mat& image, Mat& fillImage,Point seed)
{
	Mat checkMask=Mat::zeros(480,640,CV_8UC1);
	uchar* maskP=(uchar*)checkMask.data;

	uchar aaaa=image.at<uchar>(seed.y,seed.x);
	Process(fillImage,seed);
	*(maskP+seed.y*640+seed.x)=1;
	Span seedspan;
	seedspan.XLeft=seed.x;
	seedspan.XRight=seed.x;
	seedspan.Y=seed.y;
	seedspan.ParentDirection=ParentDirections::Non;
	seedspan.Extended=ExtendTypes::UnRez;
	//spanContainer.push_back(seedspan);
	spanContainer.push(seedspan);
	while(spanContainer.size()!=0)
	{
		Span span=(Span)spanContainer.top();
		//??????????????
		spanContainer.pop();
		//////////////////
		//ALLRez
		if(span.Extended==ExtendTypes::AllRez)
		{
			if(span.ParentDirection==ParentDirections::Y2)
			{
				if(span.Y-1>=0)
					CheckRange(image,fillImage,span.XLeft,span.XRight,span.Y-1,ParentDirections::Y2,checkMask);
				continue;
			}
			if(span.ParentDirection==ParentDirections::Y0)
			{
				if(span.Y+1<image.rows)
					CheckRange(image,fillImage,span.XLeft,span.XRight,span.Y+1,ParentDirections::Y0,checkMask);
				continue;
			}
		}
		if(span.Extended==ExtendTypes::UnRez)
		{
			int xl=FindXLeft(image,fillImage,checkMask,span.XLeft,span.Y);
			int xr=FindXRight(image,fillImage,checkMask,span.XRight,span.Y);
			if(span.ParentDirection==ParentDirections::Y2)
			{
				if(span.Y-1>=0)
					CheckRange(image,fillImage,xl,xr,span.Y-1,ParentDirections::Y2,checkMask);
				if(span.Y+1<image.rows)
				{
					if(xl!=span.XLeft)
						CheckRange(image,fillImage,xl,span.XLeft,span.Y+1,ParentDirections::Y0,checkMask);
					if(span.XRight!=xr)
						CheckRange(image,fillImage,span.XRight,xr,span.Y+1,ParentDirections::Y0,checkMask);
				}
				continue;
			}			
			if(span.ParentDirection==ParentDirections::Y0)
			{
				if (span.Y + 1 < image.rows)
					CheckRange(image,fillImage,xl, xr, span.Y + 1, ParentDirections::Y0,checkMask);
				if (span.Y - 1 >= 0)
				{
					if (xl != span.XLeft)
						CheckRange(image,fillImage,xl, span.XLeft, span.Y - 1, ParentDirections::Y2,checkMask);
					if (span.XRight != xr)
						CheckRange(image,fillImage,span.XRight, xr, span.Y - 1, ParentDirections::Y2,checkMask);
				}
				continue;
			}

			if (span.ParentDirection == ParentDirections::Non)
			{
				if (span.Y + 1 < image.rows)
					CheckRange(image,fillImage,xl, xr, span.Y + 1, ParentDirections::Y0,checkMask);
				if (span.Y - 1 >= 0)
					CheckRange(image,fillImage,xl, xr, span.Y - 1, ParentDirections::Y2,checkMask);
				continue;
			}
		}

		if (span.Extended == ExtendTypes::LeftRequired)
		{
			int xl = FindXLeft(image,fillImage,checkMask,span.XLeft, span.Y);
			if (span.ParentDirection == ParentDirections::Y2)
			{
				if (span.Y - 1 >= 0)
					CheckRange(image,fillImage,xl, span.XRight, span.Y - 1, ParentDirections::Y2,checkMask);
				if (span.Y + 1 < image.rows&& xl != span.XLeft)
					CheckRange(image,fillImage,xl, span.XLeft, span.Y + 1, ParentDirections::Y0,checkMask);
				continue;
			}
			if (span.ParentDirection == ParentDirections::Y0)
			{
				if (span.Y + 1 < image.rows)
					CheckRange(image,fillImage,xl, span.XRight, span.Y + 1, ParentDirections::Y0,checkMask);
				if (span.Y - 1 >= 0 && xl != span.XLeft)
					CheckRange(image,fillImage,xl, span.XLeft, span.Y - 1, ParentDirections::Y2,checkMask);
				continue;
			}

			if (span.Extended == ExtendTypes::RightRequired)
			{
				int xr = FindXRight(image,fillImage,checkMask,span.XRight, span.Y);

				if (span.ParentDirection == ParentDirections::Y2)
				{
					if (span.Y - 1 >= 0)
						CheckRange(image,fillImage,span.XLeft, xr, span.Y - 1, ParentDirections::Y2,checkMask);
					if (span.Y + 1 < image.rows && span.XRight != xr)
						CheckRange(image,fillImage,span.XRight, xr, span.Y + 1, ParentDirections::Y0,checkMask);
					continue;
				}

				if (span.ParentDirection == ParentDirections::Y0)
				{
					if (span.Y + 1 < image.rows)
						CheckRange(image,fillImage,span.XLeft, xr, span.Y + 1, ParentDirections::Y0,checkMask);
					if (span.Y - 1 >= 0 && span.XRight != xr)
						CheckRange(image,fillImage,span.XRight, xr, span.Y - 1, ParentDirections::Y2,checkMask);
					continue;
				}
			}
		}
	}

	imshow("fill",fillImage);
}


void mouseEvent(int evt, int x, int y, int flags, void* param) {                    
   std::cout<<derivY.at<short>(y,x)<<endl;
}



Point lineInterContour(vector<Point> contour,Point p1,Point p2)
{
	cv::Vec2f lineVec(p2.x-p1.x,p2.y-p1.y);
	lineVec/=100.0f;
	float interX=0;
	float interY=0;
	int interCount=0;

	for(int i=0;i<100;++i)
	{
		Point p(p1.x+lineVec(0)*i,p1.y+lineVec(1)*i);
		if(cv::pointPolygonTest(contour,p,false)>=0)
		{
			interX+=p.x;
			interY+=p.y;
			interCount++;
		}
	}

	Point interCenter(interX/interCount,interY/interCount);
	return interCenter;
}

void dilation(int dilationSize,Mat& inputMat,Mat& outputMat)
{
	cv::Mat dilationElement= getStructuringElement( MORPH_CROSS,
		Size( 2*dilationSize + 1, 2*dilationSize+1 ),
		Point( dilationSize,dilationSize) );

	dilate(inputMat, outputMat, dilationElement );
}

void cannyDetector(Mat& image,Mat& edgeImage,int lowThreshold,int ratio,int kernel_size)
{
	blur(image,edgeImage,Size(3,3));
	Canny(edgeImage,edgeImage,lowThreshold,lowThreshold*ratio,kernel_size);
	imshow("Edges",edgeImage);
}

void calculateHistogram(Mat& image,int histSize,float range1,float range2,int hist_w,int hist_h)
{
	Mat depthHist;

	float range[] = { range1, range2 } ;
	const float* histRange = { range };

	int bin_w = cvRound( (double) hist_w/histSize );

	Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );

	Mat aaa=Mat::zeros(480,640,CV_8UC1);
	cv::calcHist(&image,1,0,Mat(),depthHist,1,&histSize,&histRange,true,false);

	normalize(depthHist,depthHist,0,histImage.rows,NORM_MINMAX, -1, Mat());

	for( int i = 1; i < histSize; i++ )
	{
		line( histImage, Point( bin_w*(i-1), hist_h - cvRound(depthHist.at<float>(i-1)) ) ,
			Point( bin_w*(i), hist_h - cvRound(depthHist.at<float>(i)) ),
			Scalar( 255, 0, 0), 2, 8, 0  );
	}

	imshow("histogram",histImage);
}

void setPositive( int value, void* )
{
	positiveValue=value;
}

void setNegative( int value, void* )
{
	negativeValue=value;
}

void setminCr( int value, void* )
{
	minCr=value;
}

void setmaxCr( int value, void* )
{
	maxCr=value;
}

void setminCb( int value, void* )
{
	minCb=value;
}

void setmaxCb( int value, void* )
{
	maxCb=value;
}

void setminY( int value, void* )
{
	minY=value;
}

void setmaxY( int value, void* )
{
	maxY=value;
}



void setErosion( int value, void* )
{
	erosionSize=value;
}

void setDilation( int value, void* )
{
	dilationSize=value;
}

void erosion(int erosionSize,Mat& inputMat,Mat& outputMat)
{
	cv::Mat erosionElement= getStructuringElement(MORPH_CROSS,
		Size( 2*erosionSize + 1, 2*erosionSize+1 ),
		Point( erosionSize, erosionSize ) );

	erode(inputMat,outputMat,erosionElement);
}



void backGroundSubstraction(Mat& backGroundImage, Mat& capturedImage,int dilationSize,int erosionSize,Mat& foreGroundMask, Mat& foreGroundImage )
{

	Mat backgroundYCrCb;
	cv::cvtColor(backGroundImage,backgroundYCrCb,cv::COLOR_BGR2YCrCb);

	Mat imageYCrCbOne;
	cv::cvtColor(capturedImage,imageYCrCbOne,cv::COLOR_BGR2YCrCb);

	//Calculation the substraction between two images in the space of YCrCb
	Mat differenceYCrCb;
	cv::absdiff(backgroundYCrCb,imageYCrCbOne,differenceYCrCb);

	Mat imageYCrCbChannels[3];
	split(imageYCrCbOne,imageYCrCbChannels);

	Mat differenceYCrCbChannels[3];
	split(differenceYCrCb, differenceYCrCbChannels);

	//Get three channels of the difference image
	uchar *pixelDifference1=(uchar*)differenceYCrCbChannels[0].data;
	uchar *pixelDifference2=(uchar*)differenceYCrCbChannels[1].data;
	uchar *pixelDifference3=(uchar*)differenceYCrCbChannels[2].data;

	//Get three channels of the captured image
	uchar *pixelY=(uchar*)imageYCrCbChannels[0].data;
	uchar *pixelCr=(uchar*)imageYCrCbChannels[1].data;
	uchar *pixelCb=(uchar*)imageYCrCbChannels[2].data;

	createTrackbar( "minY:\n 0:", "Y Channel",
		&minY, 255,
		setminY );
	createTrackbar( "maxY:\n 2n +1", "Y Channel",
		&maxY, 255,
		setmaxY );
	createTrackbar( "minCr:\n 0: ", "Cr Channel",
		&minCr, 255,
		setminCr );
	createTrackbar( "maxCr:\n 2n +1", "Cr Channel",
		&maxCr, 255,
		setmaxCr );
	createTrackbar( "minCb:\n 0: ", "Cb Channel",
		&minCb, 255,
		setminCb );
	createTrackbar( "maxCb:\n 2n +1", "Cb Channel",
		&maxCb, 255,
		setmaxCb );
	createTrackbar( "erosion:\n 0: ", "foreGround",
		&erosionSize, 10,
		setErosion );
	createTrackbar( "dilation:\n 2n +1", "foreGround",
		&dilationSize, 10,
		setDilation );

	for(int i=0;i<640*480;++i)
	{
		if(*(pixelDifference1+i)<minY||*(pixelDifference1+i)>maxY)
			*(pixelY+i)=0;

		if(*(pixelDifference2+i)<minCr||*(pixelDifference2+i)>maxCr)
			*(pixelCr+i)=0;

		if(*(pixelDifference3+i)<minCb||*(pixelDifference3+i)>maxCb)
			*(pixelCb+i)=0;
	}

	//Make dilation on three channels
	dilation(dilationSize,imageYCrCbChannels[0],imageYCrCbChannels[0]);
	dilation(dilationSize,imageYCrCbChannels[1],imageYCrCbChannels[1]);
	dilation(dilationSize,imageYCrCbChannels[2],imageYCrCbChannels[2]);

	//Make erosion on three channels
	erosion(erosionSize,imageYCrCbChannels[0],imageYCrCbChannels[0]);
	erosion(erosionSize,imageYCrCbChannels[1],imageYCrCbChannels[1]);
	erosion(erosionSize,imageYCrCbChannels[2],imageYCrCbChannels[2]);

	//Turn three channels into masks
	cv::threshold(imageYCrCbChannels[0],imageYCrCbChannels[0],0,255,0);
	cv::threshold(imageYCrCbChannels[1],imageYCrCbChannels[1],0,255,0);
	cv::threshold(imageYCrCbChannels[2],imageYCrCbChannels[2],0,255,0);

	uchar* foreMaskPixel=(uchar*)foreGroundMask.data;
	uchar* foreGroundPixel=(uchar*)foreGroundImage.data;
	uchar* rawColorPixel=(uchar*)capturedImage.data;

	for(int i=0;i<480*640;++i)
	{
		if(i>400*640)
		break;

		if(*pixelY!=0||*pixelCr!=0||*pixelCb!=0)
		{
			*foreMaskPixel=255;
			*foreGroundPixel=*rawColorPixel;
			*(foreGroundPixel+1)=*(rawColorPixel+1);
			*(foreGroundPixel+2)=*(rawColorPixel+2);
		}
		pixelY++;
		pixelCr++;
		pixelCb++;
		foreMaskPixel++;
		foreGroundPixel+=3;
		rawColorPixel+=3;
	}

	imshow("Y Channel",imageYCrCbChannels[0]);
	imshow("Cr Channel",imageYCrCbChannels[1]);
	imshow("Cb Channel",imageYCrCbChannels[2]);
	imshow("ForeGroundMask",foreGroundMask);
    imshow("foreGround",foreGroundImage);
}

void filterDepthThreshold(Mat& rawDepth,Mat&filterDepth,int threshold)
{
	unsigned short * rawDepthP=(unsigned short*)rawDepth.data;
	unsigned short * filterP=(unsigned short*)filterDepth.data;

	//Filter depth larger than 800
	for(int i=0;i<640*480;++i)
	{
		if(*(rawDepthP+i)<threshold&&*(rawDepthP+i)>0)
		{
			*(filterP+i)=*(rawDepthP+i);
		}
		else
			*(filterP+i)=0;
	}
}

//Calculate the world coordinate of a 2D point in the depth image
cv::Matx31f DepthToWorld(int x, int y, int depthValue,int cameraIndex)
{
	double fx_d;
	double fy_d;
	double cx_d;
	double cy_d;

	if(cameraIndex==1)
	{
		fx_d=1.0/585.30;
		fy_d=1.0/582.11;
		cx_d=314.91;
		cy_d=244.17;
	}

	if(cameraIndex==2)
	{
		fx_d=1.0/584.92;
		fy_d=1.0/580.90;
		cx_d=311.26;
		cy_d=247.24;
	}

	float newDepth=(float)depthValue/1000;
	cv::Matx31f result(float((x-cx_d)*newDepth*fx_d),float((y-cy_d)*newDepth*fy_d),float(newDepth));
	return result;
}

//Transform the world coordinate of the depth reference to the color reference
cv::Matx31f DepthWorldToColorWorld(const cv::Matx31f & pt,int cameraIndex)
{
	cv::Matx33f rotationMatrix;
	cv::Matx31f translation;

	if(cameraIndex==1)
	{
		cv::Matx33f rotationMatrixOne( 0.99991,-0.00994, 0.00939,
		                               0.00999, 0.99994,-0.00447,
						              -0.00934, 0.00456, 0.99995);
		rotationMatrix=rotationMatrixOne;

		cv::Matx31f translationOne(-0.03039,0.00043,-0.00283);
		translation=translationOne;
	}
	if(cameraIndex==2)
	{
		cv::Matx33f rotationMatrixTwo( 0.99999,-0.00439, 0.00137,
		                               0.00440, 0.99997,-0.00665,
						              -0.00134, 0.00665, 0.99998);
		rotationMatrix=rotationMatrixTwo;

		cv::Matx31f translationTwo(-0.01605,0.00076,-0.00125);
		translation=translationTwo;
	}
	
	cv::Matx31f transformedPos;
	transformedPos=rotationMatrix*pt+translation;
	return transformedPos;
}

void WorldToColor(const cv::Matx31f & transformedPos,int cameraIndex,cv::Matx21f& undisResult)
{
	double fx_rgb;
	double fy_rgb;
	double cx_rgb;
	double cy_rgb;

	if(cameraIndex==1)
	{
		fx_rgb=1048.32/2;
		fy_rgb=1044.99/2;
		cx_rgb=636.90/2;
		cy_rgb=549.48/2;
	}
	if(cameraIndex==2)
	{
		fx_rgb=1062.87/2;
		fy_rgb=1061.18/2;
		cx_rgb=639.73/2;
		cy_rgb=537.50/2;
	}
	
	const float invZ=1.0f/transformedPos(2);
	
	cv::Matx21f normalized;

	normalized(0)=transformedPos(0)*invZ;
	normalized(1)=transformedPos(1)*invZ;

	undisResult(0)=int((normalized(0)*fx_rgb)+cx_rgb+0.5);
	undisResult(1)=int((normalized(1)*fy_rgb)+cy_rgb+0.5);
}

//Register the depth image to the color image
void setCorrectedDepthImage(const cv::Mat& rawDepth, cv::Mat& undisCorrectedDepth,cv::Mat& undisNormalizedDepth, int cameraIndex)
{
	unsigned short* rawP=(unsigned short*)rawDepth.data;
	unsigned short* undisCorrectP=(unsigned short*)undisCorrectedDepth.data;
	uchar* undisNormalizeP=(uchar*)undisNormalizedDepth.data;

	for(int j=0;j<rawDepth.rows;j++)
	{

		for(int i=0;i<rawDepth.cols;i++)
		{

			int value=*(rawP+j*640+i);
			if(value==0)
				continue;

			//Calculate the coordinate of each pixel of the depth image 
			cv::Matx31f pixelWorld=DepthToWorld(i,j,value,cameraIndex);
			cv::Matx31f transformedPos=DepthWorldToColorWorld(pixelWorld,cameraIndex);
			/*cv::Matx21f depthRegistered=WorldToColor(transformedPos,cameraIndex);*/
			cv::Matx21f undisDepthRegistered;
			WorldToColor(transformedPos,cameraIndex,undisDepthRegistered);
			//
			//Filter out the point which is outside the size of the image 640*480
			if(undisDepthRegistered(0)>=0&&undisDepthRegistered(0)<640&&undisDepthRegistered(1)>=0&&undisDepthRegistered(1)<480)
			{
				if(*(undisCorrectP+(int)undisDepthRegistered(1)*640+(int)undisDepthRegistered(0))!=0&&*(undisCorrectP+(int)undisDepthRegistered(1)*640+(int)undisDepthRegistered(0))<value)
					continue;
				*(undisCorrectP+(int)undisDepthRegistered(1)*640+(int)undisDepthRegistered(0))=value;
				*(undisNormalizeP+(int)undisDepthRegistered(1)*640+(int)undisDepthRegistered(0))=value*255/10000;
			}
		}
	}
}

void findSlice(Mat& image,vector<vector<Point>>slices,int positiveThreshold,int negativeThreshold)
{
	slices.clear();
	slices.resize(640);

	Mat sliceImage=Mat::zeros(480,640,CV_8UC1);

    short* imageP=(short*)image.data;
	uchar* sliceP=(uchar*)sliceImage.data;

	int upCount=0;
	int flatCount=0;
	int downCount=0;
	int downDropCount=0;
	int endCount=0;
	
	for(int i=0;i<640;++i)
	{
		bool sliceUp=false;
		bool sliceFlat=false;
		bool sliceDown=false;
		bool sliceEnd=false;

		int startIndex=-1;

		vector<Point> slice;

		bool drop=false;

		for(int j=0;j<480;++j)
		{
			short derivative=*(imageP+j*640+i);
			Point point(i,j);

			if(sliceUp==false)
			{
				if(derivative>positiveThreshold)
				{
					sliceUp=true;		
					slice.push_back(point);
					startIndex=j;
					upCount++;
					//*(sliceP+j*640+i)=255;
				}
			}
			else
			{
				/*if((j-startIndex)>200)
					break;*/

				if(sliceEnd==true)
				{
					for(int k=0;k<slice.size();++k)
					{
						slices[i].push_back(slice[k]);
					}	
					break;
				}

				if(sliceFlat==false)
				{
					//if(derivative<negativeThreshold)
					//	break;

					if(derivative>positiveThreshold)
					{	
						slice.push_back(point);
						//*(sliceP+j*640+i)=255;
					}

					if(derivative>negativeThreshold&&derivative<positiveThreshold)
					{
						
						sliceFlat=true;
						slice.push_back(point);
						flatCount++;
						//*(sliceP+j*640+i)=255;
					}
				}
				else
				{
					if(sliceDown==false)
					{
					/*	if(derivative>positiveThreshold)
						{
							downDropCount++;
							break;
						}*/
							
						/*if(derivative>positiveThreshold)
						{
							if(drop==false)
							{
								downDropCount++;
								drop=true;
							}			
						}*/

						if(derivative>negativeThreshold/*&&derivative<positiveThreshold*/)
						{
							slice.push_back(point);
							//*(sliceP+j*640+i)=255;
						}

						if(derivative<negativeThreshold)
						{
							sliceDown=true;
							slice.push_back(point);
							downCount++;
							//*(sliceP+j*640+i)=255;
						}
					}

					else
					{
						//if(derivative>positiveThreshold)
						//	break;

						if(derivative<negativeThreshold)
						{
							slice.push_back(point);
							//*(sliceP+j*640+i)=255;
						}
						if(derivative>negativeThreshold&&derivative<positiveThreshold)
						{
							endCount++;
							sliceEnd=true;
						}
					}
				}		
			}			
		}
	}

	std::cout<<"Up: "<<upCount<<"  Flat: "<<flatCount<<"  Down: "<<downCount<<"  DownDrop: "<<downDropCount<<"  end: "<<endCount<<endl;

	for(int i=0;i<slices.size();++i)
	{
		for(int j=0;j<slices[i].size();++j)
		{
			*(sliceP+slices[i][j].y*640+slices[i][j].x)=255;
		}
		
	}
	imshow("slice",sliceImage);
}
int _tmain(int argc, _TCHAR* argv[])
{
	 OpenNI::initialize();

    Array<DeviceInfo> aDeviceList;
	OpenNI::enumerateDevices(&aDeviceList);

	Device deviceOne;


	
	VideoMode mModeDepth;
    mModeDepth.setResolution( 640, 480 ); 
    mModeDepth.setFps( 30 );
    mModeDepth.setPixelFormat( PIXEL_FORMAT_DEPTH_1_MM );

	VideoMode mModeColor;
	mModeColor.setResolution( 640, 480 );
    mModeColor.setFps( 30 );
    mModeColor.setPixelFormat( PIXEL_FORMAT_RGB888 );

    //Camera 1
	deviceOne.open(aDeviceList[0].getUri());

    VideoStream streamDepthOne;
    streamDepthOne.create( deviceOne, SENSOR_DEPTH );
	
	VideoStream streamColorOne;
    streamColorOne.create( deviceOne, SENSOR_COLOR );

	streamDepthOne.setVideoMode( mModeDepth);
	streamColorOne.setVideoMode( mModeColor);

	streamDepthOne.start();
	streamColorOne.start();

	int iMaxDepthOne = streamDepthOne.getMaxPixelValue();

	VideoFrameRef  frameDepthOne;
	VideoFrameRef  frameColorOne;

	while(true)
	{
		streamDepthOne.readFrame( &frameDepthOne );
        streamColorOne.readFrame( &frameColorOne );

		//Get raw depth image
        const cv::Mat mImageDepthOne( frameDepthOne.getHeight(), frameDepthOne.getWidth(), CV_16UC1, (void*)frameDepthOne.getData());
        
		//Mirror the image
		cv::flip(mImageDepthOne,mImageDepthOne,1);

		//Get raw color image
		const cv::Mat mImageRGBOne(frameColorOne.getHeight(), frameColorOne.getWidth(), CV_8UC3, (void*)frameColorOne.getData());


		cv::Mat cImageBGROne;
		cv::Mat cImageBGRTwo;

		cv::cvtColor(mImageRGBOne,cImageBGROne,CV_RGB2BGR);
		cv::flip(cImageBGROne,cImageBGROne,1);


		cv::Mat cImageGRAYOne;
		cv::cvtColor(mImageRGBOne,cImageGRAYOne,CV_BGR2GRAY);

		//Set corner points
		tableCorners[0]=Point(104,105);
		tableCorners[1]=Point(538,105);
		tableCorners[2]=Point(539,353);
		tableCorners[3]=Point(98,350);

		for(int i=0;i<4;++i)
		{
			cv::circle(cImageBGROne,tableCorners[i],2,Scalar(0,0,255));

		}

		imshow( "ColorOne", cImageBGROne );
		//cv::imwrite("colorBackground1.jpg",cImageBGROne);


		//Convert to depth image which can be displayed
        cv::Mat mScaledDepthOne;
        mImageDepthOne.convertTo( mScaledDepthOne, CV_8U, 255.0 / iMaxDepthOne );

		cv::Mat testUndisCorrectedOne=cv::Mat::zeros(480,640,CV_16UC1);
		cv::Mat newUndisDepthOne=cv::Mat::zeros(480,640,CV_8UC1);
		cv::Mat testDisCorrectedOne=cv::Mat::zeros(480,640,CV_16UC1);
		cv::Mat newDisDepthOne=cv::Mat::zeros(480,640,CV_8UC1);

		setCorrectedDepthImage(mImageDepthOne,testUndisCorrectedOne,newUndisDepthOne,1);

		imshow("UndisDepth1",newUndisDepthOne);
		
		//Filter the depth image by using a depth threshold value
		Mat filterDepth=Mat::zeros(480,640,CV_16UC1);
		filterDepthThreshold(testUndisCorrectedOne,filterDepth,1000/*1200*/);
		Mat filterNormalizeDepth;
		filterDepth.convertTo(filterNormalizeDepth,CV_8U,255.0 /1000 /*1200*/);
		Mat maskHand;
		cv::threshold(filterNormalizeDepth,maskHand,1,255,0);
		//cv::imwrite("hand.bmp",maskHand);
		
		//Calculate and draw the histogram
		calculateHistogram(filterDepth,512,1,1000,1000,400);
		
		Mat edges=Mat::zeros(480,640,CV_8UC1);
		cannyDetector(filterNormalizeDepth,edges,100,3,5);
		
		Mat dilationEdges=Mat::zeros(480,640,CV_8UC1);
		dilation(1,edges,dilationEdges);

		vector<vector<Point> > contours;
		vector<Vec4i> hierarchy;

		cv::findContours(dilationEdges,contours,CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

		Mat colorContour=Mat::zeros(480,640,CV_8UC3);
		cv::line(colorContour,tableCorners[0],tableCorners[1],Scalar(0,0,255));
		cv::line(colorContour,tableCorners[1],tableCorners[2],Scalar(0,0,255));
		cv::line(colorContour,tableCorners[2],tableCorners[3],Scalar(0,0,255));
		cv::line(colorContour,tableCorners[3],tableCorners[0],Scalar(0,0,255));
		for(int i=0;i<contours.size();++i)
		{
			cv::drawContours(colorContour,contours,i,Scalar(rng.uniform(0, 255),rng.uniform(0, 255),rng.uniform(0, 255)));
			Point interCenter(-1,-1);
			interCenter=lineInterContour(contours[i],tableCorners[0],tableCorners[1]);
			std::cout<<interCenter.x<<" "<<interCenter.y;
			//Mat fillImage=Mat::zeros(480,640,CV_8UC1);
			//if(interCenter.x>=0&&interCenter.x<640&&interCenter.y>=0&&interCenter.y<480)
			//ExcuteSpanFill(maskHand,fillImage,interCenter);
	     	cv::circle(colorContour,interCenter,2,Scalar(0,255,0));
		}
		//cv::imwrite("colorhand.jpg",colorContour);
		Mat fillImage=Mat::zeros(480,640,CV_8UC1);
		Mat hand=cv::imread("hand.bmp");
		Point handCenter(277,105);
		ExcuteSpanFill(hand,fillImage,handCenter);

		imshow("colorContour",colorContour);


		imshow("MaskHand",maskHand);
		imshow("FilterDepth",filterNormalizeDepth);
		imshow("dilation",dilationEdges);

		/*Mat derivX;
		Mat derivY;
		Mat deriv;*/
		cv::Sobel(filterDepth,derivX,CV_16UC1,1,0,5,1,0,BORDER_DEFAULT);
		cv::Sobel(filterDepth,derivY,CV_16UC1,0,1,5,1,0,BORDER_DEFAULT);




		/*std::cout<<"min :"<<min<<"   max: "<<max<<endl;*/
		
		cv::addWeighted(derivX,0.5,derivY,0.5,0,deriv);

		createTrackbar("positiveValue:\n 0:", "slice",
					&positiveValue, 500,
					setPositive);
		createTrackbar("negativeValue:\n 1:", "slice",
					&negativeValue, 500,
					setNegative);

		

		/*Mat derivXNormal;
		Mat derivYNormal;
		Mat derivNormal;*/

		
		convertScaleAbs(derivX, derivXNormal);
		convertScaleAbs(derivY, derivYNormal);
		convertScaleAbs(deriv, derivNormal);

		imshow("derivXNormal",derivX);
		imshow("derivYNormal",derivY);
		imshow("derivNormal",derivNormal);

		cvSetMouseCallback("derivYNormal", mouseEvent, &derivY);

		if(cv::waitKey(1)=='s')
		{

		}
		if(cv::waitKey(1)=='q')
			break;
	}

	streamDepthOne.destroy();
	streamColorOne.destroy();
	deviceOne.close();
	return 0;
}

