#define _CRT_SECURE_NO_WARNINGS
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <malloc.h>
#include <vector>
#include <fstream>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

typedef struct {
	int r, g, b;
}int_rgb;

#define SQ(x) ((x)*(x))

void WriteToCSV(char* name, int* data, int num)
{
	Mat src(num, 1, CV_32SC1);
	for (int i = 0; i < num; i++)
		src.at<int>(i, 0) = data[i];

	std::ofstream fs;
	fs.open(name);
	fs << format(src, Formatter::FMT_CSV) << std::endl;
	fs.close();
}

int** IntAlloc2(int height, int width)
{
	int** tmp;
	tmp = (int**)calloc(height, sizeof(int*));
	for (int i = 0; i<height; i++)
		tmp[i] = (int*)calloc(width, sizeof(int));
	return(tmp);
}

void IntFree2(int** image, int height, int width)
{
	for (int i = 0; i<height; i++)
		free(image[i]);

	free(image);
}


float** FloatAlloc2(int height, int width)
{
	float** tmp;
	tmp = (float**)calloc(height, sizeof(float*));
	for (int i = 0; i<height; i++)
		tmp[i] = (float*)calloc(width, sizeof(float));
	return(tmp);
}

void FloatFree2(float** image, int height, int width)
{
	for (int i = 0; i<height; i++)
		free(image[i]);

	free(image);
}

int_rgb** IntColorAlloc2(int height, int width)
{
	int_rgb** tmp;
	tmp = (int_rgb**)calloc(height, sizeof(int_rgb*));
	for (int i = 0; i<height; i++)
		tmp[i] = (int_rgb*)calloc(width, sizeof(int_rgb));
	return(tmp);
}

void IntColorFree2(int_rgb** image, int height, int width)
{
	for (int i = 0; i<height; i++)
		free(image[i]);

	free(image);
}

int** ReadImage(char* name, int* height, int* width)
{
	Mat img = imread(name, IMREAD_GRAYSCALE);
	int** image = (int**)IntAlloc2(img.rows, img.cols);

	*width = img.cols;
	*height = img.rows;

	for (int i = 0; i<img.rows; i++)
		for (int j = 0; j<img.cols; j++)
			image[i][j] = img.at<unsigned char>(i, j);

	return(image);
}

void WriteImage(char* name, int** image, int height, int width)
{
	Mat img(height, width, CV_8UC1);
	for (int i = 0; i<height; i++)
		for (int j = 0; j<width; j++)
			img.at<unsigned char>(i, j) = (unsigned char)image[i][j];

	imwrite(name, img);
}


void ImageShow(char* winname, int** image, int height, int width)
{
	Mat img(height, width, CV_8UC1);
	for (int i = 0; i<height; i++)
		for (int j = 0; j<width; j++)
			img.at<unsigned char>(i, j) = (unsigned char)image[i][j];
	imshow(winname, img);
	waitKey(0);
}



int_rgb** ReadColorImage(char* name, int* height, int* width)
{
	Mat img = imread(name, IMREAD_COLOR);
	int_rgb** image = (int_rgb**)IntColorAlloc2(img.rows, img.cols);

	*width = img.cols;
	*height = img.rows;

	for (int i = 0; i<img.rows; i++)
		for (int j = 0; j < img.cols; j++) {
			image[i][j].b = img.at<Vec3b>(i, j)[0];
			image[i][j].g = img.at<Vec3b>(i, j)[1];
			image[i][j].r = img.at<Vec3b>(i, j)[2];
		}

	return(image);
}

void WriteColorImage(char* name, int_rgb** image, int height, int width)
{
	Mat img(height, width, CV_8UC3);
	for (int i = 0; i<height; i++)
		for (int j = 0; j < width; j++) {
			img.at<Vec3b>(i, j)[0] = (unsigned char)image[i][j].b;
			img.at<Vec3b>(i, j)[1] = (unsigned char)image[i][j].g;
			img.at<Vec3b>(i, j)[2] = (unsigned char)image[i][j].r;
		}

	imwrite(name, img);
}

void ColorImageShow(char* winname, int_rgb** image, int height, int width)
{
	Mat img(height, width, CV_8UC3);
	for (int i = 0; i<height; i++)
		for (int j = 0; j<width; j++) {
			img.at<Vec3b>(i, j)[0] = (unsigned char)image[i][j].b;
			img.at<Vec3b>(i, j)[1] = (unsigned char)image[i][j].g;
			img.at<Vec3b>(i, j)[2] = (unsigned char)image[i][j].r;
		}
	imshow(winname, img);

	waitKey(0);

}

void DrawHistogram(char* comments, int* Hist)
{
	int histSize = 256; /// Establish the number of bins
						// Draw the histograms for B, G and R
	int hist_w = 512; int hist_h = 512;
	int bin_w = cvRound((double)hist_w / histSize);

	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(255, 255, 255));
	Mat r_hist(histSize, 1, CV_32FC1);
	for (int i = 0; i < histSize; i++)
		r_hist.at<float>(i, 0) = Hist[i];
	/// Normalize the result to [ 0, histImage.rows ]
	normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	/// Draw for each channel
	for (int i = 1; i < histSize; i++)
	{
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(r_hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(r_hist.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);
	}

	/// Display
	namedWindow(comments, WINDOW_AUTOSIZE);
	imshow(comments, histImage);

	waitKey(0);

}

template <typename _TP>
void ConnectedComponentLabeling(_TP** seg, int height, int width, int** label, int* no_label)
{

	//Mat bw = threshval < 128 ? (img < threshval) : (img > threshval);
	Mat bw(height, width, CV_8U);

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++)
			bw.at<unsigned char>(i, j) = (unsigned char)seg[i][j];
	}
	Mat labelImage(bw.size(), CV_32S);
	*no_label = connectedComponents(bw, labelImage, 8); 

	(*no_label)--;

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++)
			label[i][j] = labelImage.at<int>(i, j);
	}
}   

//이미지 불러오고 띄우는 함수
void main_1stClass()
{	//Mat : 클래스, img : 오브젝트
	Mat img = imread("barbara.png", IMREAD_GRAYSCALE);
	imshow("바바라영상", img);
	//waitKey(1000); //1000ms = 1초동안화면 정지  
	waitKey(0); //->키보드를 누르기 전까지 계속 화면 띄우기
}


//이미지 클리핑 함수
int Clipping(int pixel) 
{
	if (pixel > 255) //8비트를 넘어가는 부분 밝기 보정 하는 코드
		pixel = 255;
	if (pixel < 0) // 음수 보정하기
		pixel = 0;
	return(pixel);

	
}

// 이미지 복사 함수
void CopyImage(int**src, int h, int w, int**dst)
{
	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			dst[y][x] = src[y][x];
		}
	}
}

//전체적으로 밝기 조절 함수
void ShiftImage(int value, int** img, int height, int width, int** img_out) // 함수로 만들기
{
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			img_out[y][x] = img[y][x] + value; //이미지는 8비트로 구성. value을 +(밝아짐) / -(어두워짐)
											   //8 비트를 넘을 때가 생김->이때 이미지는 맨 앞을 버림
											   //100110010 --> 00110010 으로 됨
											   // 49-50 과같은 음수가 나오면 2의 보수를 사용하기 때문에 11111111 --> 양수로 취급하면 굉장히 밝음 
											   //- 해결하는것도 + 때와 비슷하게 해결하기-->클러핑 함수 사용 해야함!!!!!
			img_out[y][x] = Clipping(img_out[y][x]);//클리핑 함수 사용
		}
	}
}


// 상자 부분만 밝기조절 방법1,방법2
void ShiftImage_2(int value, int y_st, int x_st, int y_end, int x_end, int** img, int height, int width, int** img_out) // 함수로 만들기
{
	// 방법1
/*
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			//박스 내부
			if ((x_st < x && x < x_end) && (y_st < y && y < y_end))
			{
				img_out[y][x] = img[y][x] + value;
				img_out[y][x] = Clipping(img_out[y][x]);//클리핑 함수 사용
			}
			//박스 외부
			else {
				img_out[y][x] = img[y][x];
			}
		}
	}
*/

	// 방법2
	//이미지 복사하기 --> 함수로 만듬
	
	CopyImage(img, height, width,img_out); 

	//사각형 영역에만 value 더해줌
	for (int y = y_st; y < y_end; y++) {
		for (int x = x_st; x < x_end; x++) {
			//img_out[y][x] = img[y][x] + value;
			//img_out[y][x] = Clipping(img_out[y][x]);//클리핑 함수 사용
			img_out[y][x] = Clipping(img_out[y][x]+value); // 위 두줄을 한줄로 만들기
		}
	}
}

// 일차함수 부분만 밝기조절
// ********9월 10일 과제**********
void p0910_homework1(float a, float b, int value, int**img, int height, int width, int**img_out)
{
	int x,y;
	CopyImage(img, height, width, img_out);
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			if (y >= a*x + b) {
				img_out[y][x] = img[y][x] + value;
				img_out[y][x] = Clipping(img_out[y][x]);
			}
		}
	}
}
void p0910_homework2(float a, float b, int value, int**img, int height, int width, int**img_out)
{
	int x, y;
	
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) 
		{
			if (y >= a*x + b) 
			{
				img_out[y][x] = img[y][x] + value;
				img_out[y][x] = Clipping(img_out[y][x]);
			}
			else img_out[y][x] = img[y][x];
		}
	}
}

struct MyCircle {
	float radius;//반경
	int c_y;// y좌표
	int c_x;// x좌표
	int thicknees;//원 두께
};

struct MyImage {
	float height;
	float width;// 
	int **img;// 
}; 

// 범위를 원으로 하기
void DrawCircle(float brightness, float radius ,  int c_y, int c_x, int thicknees, int**img, int height, int width, int**img_out)
{
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++)
		{
			float r1 = radius - (float)thicknees / 2.0;
			float r2 = radius + (float)thicknees / 2.0;
			float left = (x - c_x)*(x - c_x) + (y - c_y) *(y - c_y);
			if (left > r1*r1&& left< r2*r2)
				img_out[y][x] = brightness;
			else 
				img_out[y][x] = img[y][x];
		}
	}
}

//구조체 사용함
void DrawCircle_2(float brightness,MyCircle circle, 
	int**img, int height, int width, int**img_out)
{
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++)
		{
			float r1 = circle.radius - (float)circle.thicknees / 2.0;
			float r2 = circle.radius + (float)circle.thicknees / 2.0;
			float left = (x - circle.c_x)*(x - circle.c_x) + (y - circle.c_y) *(y - circle.c_y);
			if (left > r1*r1&& left< r2*r2)
				img_out[y][x] = brightness;
			else
				img_out[y][x] = img[y][x];
		}
	}
}

//구조체 두번 사용
void DrawCircle_3(float brightness, MyCircle circle,
	MyImage image, int**img_out)
{
	for (int y = 0; y < image.height; y++) {
		for (int x = 0; x < image.width; x++)
		{
			float r1 = circle.radius - (float)circle.thicknees / 2.0;
			float r2 = circle.radius + (float)circle.thicknees / 2.0;
			float left = (x - circle.c_x)*(x - circle.c_x) + (y - circle.c_y) *(y - circle.c_y);
			if (left > r1*r1&& left< r2*r2)
				img_out[y][x] = brightness;
			else
				img_out[y][x] = image.img[y][x];
		}
	}
}
void Binarization(int threshold, int** img, int height, int width, int** img_out)
{
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			if (img[y][x] > threshold)
				img_out[y][x] = 255;
			else
				img_out[y][x] = 0;
		}
	}
}
void SelectiveBinarization(int th1, int th2, int** img, int height, int width, int** img_out)
{
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) 
		{
			/*
			if (th1<img[y][x] && img[y][x]<th2)
				img_out[y][x] = img[y][x];
			else if (th1 >= img[y][x])
				img_out[y][x] = 0;
			else if (th2 <= img[y][x])
				img_out[y][x] = 255;
*/
			if (img[y][x] > th2) img_out[y][x] = 255;

			else if (th1 < img[y][x]) img_out[y][x] = 0;
			
			else 
				img_out[y][x] = img[y][x];
		}
	}
}

void main_EX0910()// 사각형/일차함수
{
	//영상신호와 배열 : img[y=행][x=열]
	int height, width;
	int** img = ReadImage("barbara.png", &height, &width); //img[y][x]
	int** img_out = (int**)IntAlloc2(height, width); //배열 생성
	int** img_out_2 = (int**)IntAlloc2(height, width);
	int** img_out_3 = (int**)IntAlloc2(height, width);

	ShiftImage_2(70, height/4, width/4, height*3/4, width*3/4, img, height, width, img_out ); // 함수로 만들기
	ShiftImage_2(70, height / 8, width / 2, height * 5 / 8, width, img_out, height, width, img_out_2);

	p0910_homework1(-1 , height / 4, -100, img, height, width, img_out_3);

	ImageShow("바바라영상_입력", img, height, width); //"" 안이 똑같으면 한 화면만 발생
	ImageShow("바바라영상_출력1", img_out, height, width);
	ImageShow("바바라영상_출력2", img_out_2, height, width); 
	ImageShow("바바라영상_출력3", img_out_3, height, width);

	// 만약 출력 화면이 검정색으로 되었다면 너무 밝아져서 8비트를 넘어서 어둡게 된것
	// 클러핑을 해주면 검정색으로 나타나는 부분 방지.
}

void main_EX0915() // 범위를 원으로+구조체 1번 사용_EX0915
{
	//영상신호와 배열 : img[y=행][x=열]
	int height, width;
	int** img = ReadImage("barbara.png", &height, &width); //img[y][x]
	int** img_out = (int**)IntAlloc2(height, width); //배열 생성
	

	float brightness = 150;
	MyCircle cir;
	cir.c_y = 100;
	cir.c_x = 100;
	cir.thicknees = 10;
	cir.radius = 100;

	/*
	구조체 사용전
	float brightness = 100;
	int c_y = 100;
	int c_x = 100;
	int thicknees = 10;
	int radius = 100;
	DrawCircle(brightness, radius, c_y, c_x, thicknees, img, height, width, img_out);
	*/

	DrawCircle_2(brightness, cir, img, height, width, img_out);
	//cir 구조체 사용 
	ImageShow("바바라영상_입력", img, height, width); //"" 안이 똑같으면 한 화면만 발생
	ImageShow("바바라영상_출력1", img_out, height, width); 
	
}

void main_0915_3() //구조체 2번 사용
{
	MyImage input; 
	int height, width;

	input.img = ReadImage("barbara.png", &height, &width); //img[y][x]
	input.height = height;
	input.width = width;

	int** img_out = (int**)IntAlloc2(height, width); //배열 생성
	float brightness = 100;

	MyCircle cir;
	cir.c_y = 100;
	cir.c_x = 100;
	cir.thicknees = 10;
	cir.radius = 100;
	
	DrawCircle_3(brightness, cir, input, img_out);

	ImageShow("바바라영상_입력", input.img, input.height, input.width); //"" 안이 똑같으면 한 화면만 발생
	ImageShow("바바라영상_출력1", img_out, input.height, input.width);


}

void main_EX0915_3()//이진화 프로그래밍
{
	int height, width;
	int** img = ReadImage("barbara.png", &height, &width); //img[y][x]
	int** img_out = (int**)IntAlloc2(height, width); //배열 생성
	
	int threshold = 128;
	//이진화 프로그래밍

	Binarization(threshold, img, height, width, img_out);
	

	ImageShow("바바라영상_입력", img, height, width); //"" 안이 똑같으면 한 화면만 발생
	ImageShow("바바라영상_출력1", img_out, height, width);

}
void main_EX0915_4()//선택적 이진화 프로그래밍
{
	int height, width;
	int** img = ReadImage("barbara.png", &height, &width); //img[y][x]
	int** img_out = (int**)IntAlloc2(height, width); //배열 생성

	int th1 = 100;
	int th2 = 150;
		

	SelectiveBinarization(th1, th2, img, height, width, img_out);


	ImageShow("바바라영상_입력", img, height, width); //"" 안이 똑같으면 한 화면만 발생
	ImageShow("바바라영상_출력1", img_out, height, width);
}

#define X_VALUE 270 // Macro 사용법
#define Y_VALUE -10 // Macro 사용법

#define MIN(x,y) ((x < y) ? x : y) // Macro 사용법
#define MAX(x,y) ((x > y) ? x : y)
#define _CLIPPING_(x) MAX(MIN(x, 255), 0)

#define SQ(x) ((x)*(x)) //괄호가 중요함 

void main_EX0917_1()//
 {
	// Macro 사용법 : 특정한 문자가 발견되면 그 문자에 해당하는 수로 바꿔라
	// 대소를 비교하는 구문 알기 (?에 대해서)
	// if (x>y) value = x; else value = y;  ==  value = (x>y) ? x:y ;
	//int x = X_VALUE, y = Y_VALUE;
	//int x = -10;
	//clipping 연산 하고싶다.....
	//270-->255 , -10-->0
	//x = _CLIPPING_(x); //음수를 클리핑
	//x = MAX(x, 0);

	//int min_value = MIN(x, y); //최솟값 계산하는 것
	//int max_value = MAX(x, y); //최댓값 계산하는것

	//int x_2 = SQ(x-10); //만약 위에서 정의할 때 괄호 안하면 x-10*x-10으로 값이 달라짐.
    //printf("output = %d", output); 

	int x = -10, y = -100, z = 20;
	int minvalue = MIN(x,y,z);
	printf("%d", minvalue);

}

void image_add(float alpha, int** img_1, int** img_2, int height, int width, int** img_out)
{

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++)
		{
			img_out[y][x] = alpha*img_1[y][x] + (1 - alpha)*img_2[y][x];
		}
	}
}



void main_EX0917_2() // 가중치 변화에 다른 혼합 정도 변화 
{
	int height, width;
	int** img_1 = ReadImage("barbara.png", &height, &width); //img[y][x]
	int** img_2 = ReadImage("lena.png", &height, &width); 
	int** img_out=(int**)IntAlloc2(height, width); //배열 생성

	//두개의 영상을 alpha와 (1-alpha)의 가중치를 주어 혼합하는 프로그램 작성
	//프로그램 작성이 완료되면 함수화 합니다.

	image_add(0.5, img_1, img_2, height, width, img_out);
	
	ImageShow("바바라영상", img_1, height, width); 
	ImageShow("레나영상", img_2, height, width);
	ImageShow("혼합영상_출력", img_out, height, width);
}

// *********9월 22일 과제 스트래치*********

struct PARA {
	float a;
	float b;
	float c;
	float d;
};


void stretch(PARA stretch, int** img, int height, int width, int** img_out)
{
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {

			if (img[y][x] >= 0 && img[y][x] < stretch.a) 
			{
				img_out[y][x] = (stretch.b / stretch.a)*img[y][x];
				
			}
			
			else if (img[y][x] >= stretch.a && img[y][x] < stretch.c) 
			{
				img_out[y][x] = ((stretch.d- stretch.b) / (stretch.c- stretch.a))*(img[y][x] - stretch.a) + stretch.b;
				
			}
			else if (img[y][x] >= stretch.c && img[y][x] < 255)

			{
				img_out[y][x] = ((255 - stretch.d) / (255 - stretch.c))*(img[y][x] - stretch.c) + stretch.d;
				
			}
				
		}
	}
}

#define LineEq(x,x1,y1,x0,y0) (int)(((float)y1 - y0) / (x1 - x0)*(x - x0) + y0+0.5)
// 함수를 정의로 해서 만들수도있음!!!

/*
int LineEq(int x, int x1, int y1, int x0, int y0)//일차함수 만드는 함수
{
	int y = (int)(((float)y1 - y0) / (x1 - x0)*(x - x0) + y0+0.5);
	return(y);
}
*/
void Imagestretch(int a, int b , int c, int d, int** img, int height, int width, int** img_out)
{
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {

			if (img[y][x] <= a)
			{
				img_out[y][x] = LineEq(img[y][x], a, b, 0, 0);
			}

			else if (img[y][x] <= c)
			{
				img_out[y][x] = LineEq(img[y][x], c,d, a, b);

			}
			else 

			{
				img_out[y][x] = LineEq(img[y][x],255,255, c, d);

			}

		}
	}
}

void main_EX0922() // 스트레칭
{
	int height, width;
	int** img = ReadImage("barbara.png", &height, &width); //img[y][x]

	int** img_out = (int**)IntAlloc2(height, width);

	PARA st;
	st.a = 100;
	st.b = 50;
	st.c = 150;
	st.d = 235;

	//stretch(st, img, height, width, img_out);
	Imagestretch(100, 50, 150, 235, img, height, width, img_out);

	//반올림 이야기
	/*
	float a = 100.7;
	int b = (int)(a + 0.5); // 반올림 코딩 -> int 는 소수점 버린다.
	//(int) 케스트 연산자 하는이유 -> 경고 뜸 데이터 손실 될수있다고
	printf("b = %d", b);
	*/
	
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			img_out[y][x] = _CLIPPING_((int)(0.5 * img[y][x] + 100 + 0.5));
		}
	}
	

	ImageShow("바바라영상", img, height, width);
	ImageShow("스트래치영상_출력", img_out, height, width);
}



//히스토그램 : 밝기값의 빈도수
void GetHistogram(int** img, int height, int width, int* histogram )// == int histogram[256]
{
	for (int value = 0; value < 256; value++) {
		histogram[value] = 0; // 배열을 0으로 초기화.
	}
		for (int y = 0; y < height; y++) {
			for (int x = 0; x < width; x++) {
				histogram[img[y][x]]++; //밝기 값 별 배열 생성
			}
		}
	
}

//C히스토그램 : 히스토그램 배열의 누적합 배열.
void Get_C_Histogram(int** img, int height, int width, int* chistogram)// == int histogram[256]

{
	int histogram[256];
	GetHistogram(img, height, width, histogram);// 명암별 배열 불러오기

	chistogram[0] = histogram[0];//명암 0 인 값을 그대로 chisto에 넣기

	for (int n = 1; n < 256; n++) {
		chistogram[n] = chistogram[n - 1] + histogram[n]; //누적합 계산완료 (chistogram[255])
	}
}



void HistogramEqualization(int** img, int height, int width, int** img_out) // 히스토그램 정규화
{
	int chistogram[256];
	int chistogram_scale[256];
	int sum = 0;
	GetHistogram(img, height, width, chistogram); // 각각 명암별 배열 생성
	//GetHistogram(img, height, width, chistogram_scale);
	
	for (int value = 0; value < 256; value++) {

		
		sum += chistogram[value]; // 누적합

		chistogram_scale[value] = (float)sum * 255.0 / (512 * 512);
	}

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			img_out[y][x] = chistogram_scale[img[y][x]] ;
		}
	}
	
}

void HistogramEqualization_1(int** img, int height, int width, int** img_out)
{
	// 1. c_histogram 만들기
	// 2. c_histogram_scale 만들기
	// 3. img[y][x] --> img_out[y][x]으로 변환, 변환함수는 c_histogram_scale
	int chistogram[256];
	int chistogram_scale[256];

	Get_C_Histogram(img, height, width, chistogram);



	for (int n = 0; n < 256; n++) {

		chistogram_scale[n] = (int)(chistogram[n] * 255.0 / (height*width)+0.5);
	}

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			img_out[y][x] = chistogram_scale[img[y][x]];
		}
	}

}

void main_EX0924()//
{
	int height, width;
	int** img = ReadImage("barbara.png", &height, &width); //img[y][x]barbara.png

	int chistogram[256];

	//GetHistogram(img, height, width, histogram);
	//배열 이름만 넣음 ,histrogram이 주소라 괄호 안넣음

	Get_C_Histogram(img, height, width, chistogram);
	for (int value = 0; value < 256; value++)
		printf("hist[%d] = %d\n", value, chistogram[value]);

	WriteToCSV("c_hist.csv", chistogram, 256);

}

void main_EX1006() //
{
	int height, width;
	int** img = ReadImage("lenax0.5.png", &height, &width); //img[y][x]
	int** img_out = (int**)IntAlloc2(height, width);
	
	HistogramEqualization_1(img, height, width, img_out);
	//HistogramEqualization(img, height, width, img_out);
	
	ImageShow("입력영상", img, height, width);
	ImageShow("출력영상", img_out, height, width);

}

// *********1006과제*********
//3*3 평균필터구하기
void averagethree(int** img_in, int height, int width, int** img_out)
{
	int num =(3-1)/2;
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {

			if (x < num || y < num || x >= width - num || y >= height - num) //끝에 한줄씩
			{
				img_out[y][x] = img_in[y][x];
			}

			else {
				int sum = 0;
				for (int dy = -num; dy <= num; dy++) {
					for (int dx = -num; dx <= num; dx++) {
						
						sum += img_in[y + dy][x + dx];

						
						
					}
				}
				img_out[y][x] = (int)((sum / 9.0) + 0.5);//반올림
			}
		}
	}
}
//5*5 평균필터구하기
void averagefive(int** img_in, int height, int width, int** img_out)
{
	int num = (5 - 1) / 2;
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {

			if (x < num || y < num || x >= width - num || y >= height - num)
			{
				img_out[y][x] = img_in[y][x];
			}

			else {
				int sum = 0;
				for (int dy = -num; dy <= num; dy++) {
					for (int dx = -num; dx <= num; dx++) {

						sum += img_in[y + dy][x + dx];



					}
				}
				img_out[y][x] = (int)((sum / 25.0 + 0.5));
			}
		}
	}
}
//7*7 평균필터구하기
void averageseven(int** img_in, int height, int width, int** img_out)
{
	int num = (7 - 1) / 2;
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {

			if (x < num || y < num || x >= width - num || y >= height - num)
			{
				img_out[y][x] = img_in[y][x];
			}

			else {
				int sum = 0;
				for (int dy = -num; dy <= num; dy++) {
					for (int dx = -num; dx <= num; dx++) {

						sum += img_in[y + dy][x + dx];



					}
				}
				img_out[y][x] = (int)((sum / 49.0 + 0.5));
			}
		}
	}
}

// nXn 평균필터
void MeannXnFilter(int N, int** img_in, int height, int width, int** img_out)
{
	int num = (N-1)/2;
	//int sum = 0; //여기에 선언하면 안돼. 
	//(x, y) 좌표가 바뀔 때마다, sum을 다시 계산해야 하쟎아?? 그래서 sum을 계속 초기화해줘야 한다
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {

			
			if (x < num || y < num || x >= width - num || y >= height - num)
			{
					img_out[y][x] = img_in[y][x];
			}
			else {
				int sum = 0;
				for (int dy = -num; dy <= num; dy++) {
					for (int dx = -num; dx <= num; dx++) {
						
						sum += img_in[y + dy][x + dx];
						
					}
				}
				img_out[y][x] = (int)((float)sum / (N*N) + 0.5);
				
			}
			

		}
	}
	
}
#define _MIN_(x,y) ((x<y)?x:y)
#define _MAX_(x,y) ((x>y)?x:y)

void MeannXnFilterBoundary(int N, int** img_in, int height, int width, int** img_out)
{
	int num = (N - 1) / 2;
	
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {

		
			int sum = 0;
			for (int dy = -num; dy <= num; dy++) {
				for (int dx = -num; dx <= num; dx++) {
					int new_y = y + dy;
					int new_x = x + dy;
					//4줄 만들기 
					/*
					new_y = MAX(0, new_y);
					new_x = MAX(0, new_x);
					new_y = MIN(height - 1, new_y);
					new_x = MIN(width - 1, new_x);
					*/
					//2줄 만들기

					new_y = MIN(height - 1, MAX(0, new_y));
					new_x = MIN(width - 1, MAX(0, new_x));
					/*
					if (new_y < 0) new_y = 0;
					if (new_x < 0) new_x = 0;
					if (new_y > height - 1) new_y = height - 1;
					if (new_x > width - 1) new_x = width - 1;
					*/
					sum += img_in[new_y][new_x];
				}
			
			}

			img_out[y][x] = (int)((float)sum / (N*N) + 0.5);
		}
	}

}
void main_EX1008_old()
{
	int height, width;
	int** img = ReadImage("barbara.png", &height, &width); //img[y][x]
	
	int** img_out1 = (int**)IntAlloc2(height, width);
	int** img_out2 = (int**)IntAlloc2(height, width);
	int** img_out3 = (int**)IntAlloc2(height, width);
	int** img_out4 = (int**)IntAlloc2(height, width);
	
	
	MeannXnFilter(3, img, height, width, img_out1);
	MeannXnFilter(5, img, height, width, img_out2);
	MeannXnFilter(7, img, height, width, img_out3);
	MeannXnFilter(9, img, height, width, img_out4);
	

	ImageShow("입력영상", img, height, width);
	
	ImageShow("출력영상_3x3", img_out1, height, width);
	ImageShow("출력영상_5x5", img_out2, height, width);
	ImageShow("출력영상_7x7", img_out3, height, width);
	ImageShow("출력영상_9x9", img_out4, height, width);
}

void main_EX1008_new()
{
	int height, width;
	int** img = ReadImage("barbara.png", &height, &width); //img[y][x]
	int** img_out[4];
	for (int n= 0; n < 4; n++) 
		img_out[n] = (int**)IntAlloc2(height, width);
		
	for (int num = 1; num <= 4; num++)
		//MeannXnFilter(2 * num + 1, img, height, width, img_out[num - 1]); //경계가 일정함. 왜 ? 단순 복사
		MeannXnFilterBoundary(2*num+1, img, height, width, img_out[num-1]); //경계가 자연스러움
	 
	ImageShow("입력영상", img, height, width);
	for (int n = 0; n < 4; n++) {
		char win_name[100]; 
		sprintf(win_name, "%dx%d 출력영상", 2 * (n + 1) + 1, 2 * (n + 1) + 1);
		ImageShow(win_name, img_out[n], height, width);
	}

	
}

void MeanNxNFilterWithMask(float** mask, int N, int** img_in, int height, int width, int** img_out)

{
	int num = (N - 1) / 2;

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			int sum = 0;
			for (int dy = -num; dy <= num; dy++) {
				for (int dx = -num; dx <= num; dx++) {
					int new_y = y + dy;
					int new_x = x + dx;

					new_y = MIN(height - 1, MAX(0, new_y)); //자연스러운 경계
					new_x = MIN(width - 1, MAX(0, new_x));

					sum += mask[1 + dy][1 + dx] * img_in[new_y][new_x];
				}
			}
			img_out[y][x] = sum;
			//img_out[y][x] = Clipping(img_out[y][x]);
		}
	}

}

void main_EX1015()
{
	int height, width;
	int** img = ReadImage("barbara.png", &height, &width); //img[y][x]
	
	int** img_out0 = (int**)IntAlloc2(height, width);
	int** img_out1 = (int**)IntAlloc2(height, width);
	int** img_out2 = (int**)IntAlloc2(height, width);
	
	float ** mask0 = (float**)FloatAlloc2(3, 3);
	float ** mask1 = (float**)FloatAlloc2(3, 3);
	float ** mask2 = (float**)FloatAlloc2(3, 3);
	

	mask0[0][0] = 1.0 / 9.0; mask0[0][1] = 1.0 / 9.0; mask0[0][2] = 1.0 / 9.0;
	mask0[1][0] = 1.0 / 9.0;  mask0[1][1] = 1.0 / 9.0; mask0[1][2] = 1.0 / 9.0;
	mask0[2][0] = 1.0 / 9.0;  mask0[2][1] = 1.0 / 9.0; mask0[2][2] = 1.0 / 9.0;

	mask1[0][0] = 1;  mask1[0][1] = 0; mask1[0][2] = -1;
	mask1[1][0] = 1; mask1[1][1] = 0; mask1[1][2] = -1;
	mask1[2][0] = 1; mask1[2][1] = 0; mask1[2][2] = -1; // 수직경계선느낌

	mask2[0][0] = 1; mask2[0][1] = 1; mask2[0][2] = 1;
	mask2[1][0] = 0; mask2[1][1] = 0; mask2[1][2] = 0;
	mask2[2][0] = -1;  mask2[2][1] = -1; mask2[2][2] = -1; // 수평경계선느낌

	
	MeanNxNFilterWithMask(mask0, 3, img, height, width, img_out0);
	MeanNxNFilterWithMask(mask1, 3, img, height, width, img_out1);
	MeanNxNFilterWithMask(mask2, 3, img, height, width, img_out2);
	
	ImageShow("입력영상", img, height, width);
	ImageShow("출력영상0", img_out0, height, width);
	ImageShow("출력영상1", img_out1, height, width);
	ImageShow("출력영상2", img_out2, height, width);
	
}
//***************8주차****************
//x방향으로만 에지가 있음
//그라디언트 : 입체로 생각할때 x방향과 y방향의 기울기 
//그라디언트의 방향은 어떤의미? --> abs(fx) + abs(fy) = sqrt(fx*fy + fy*fy)
//각도는 안하고 크기만함.
//입력영상을 서로 빼서 절댓값 한다. 그럼 십자가 모양으로 나옴,
//그라디언트의 크기를 구하는 함수 구하기 (for + 더하기) + 경계처리

int find_max(int** img,int height, int width)
{
	//정규화 할때 사용하는 최댓값 찾기
	int max_value = -1000;
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			//if (img[y][x] > max_value) max_value = img[y][x];
			max_value = MAX(img[y][x], max_value);
		}
	}
	
	return max_value; // 리턴은 항상 for 문 밖에 있어야해 
}
int find_min(int** img, int height, int width)
{
	int min_value = 1000;
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			//if (img[y][x] < max_value) max_value = img[y][x];
			min_value = MIN(img[y][x], min_value);
		}
	}
	//printf("min_value=%d", min_value);
	return min_value;
}

int FindMaxValue(int** img, int height, int width) {
	int max_value = img[0][0];
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			
			max_value = MAX(img[y][x], max_value);
			
		}
	}
	return (max_value);
}

int FindMinValue(int** img, int height, int width) {
	int min_value = img[0][0];
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {

			min_value = MIN(img[y][x], min_value);

		}
	}
	return (min_value);
}


/*
void Xdirection(int** img, int height, int width, int** img_out) {
	//x방향 미분연산
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width-1; x++) {
			int f_x;
			if (x = 0) {
				img_out[y][x] = 0;
				//a = 0;
			}
			else
			{
				f_x = img[y] [x + 1] - img[y][x];
				img_out[y][x + 1] = abs(f_x);
			}

		}
	}
	
}

void Ydirection(int** img, int height, int width, int** img_out) {
	//y방향 미분연산
	for (int y = 0; y < height-1; y++) {
		for (int x = 0; x < width; x++) {
			int f_y;
			if (y = 0) {
				img_out[y][x] = 0;
			}
			else
			{
				f_y = img[y + 1][x] - img[y][x];
				img_out[y + 1][x] = abs(f_y);
			}
			
		}
	}
}


void MagGradient_fail(int** img, int height, int width, int** img_out)//경계값에 단순 0을 넣는 방법
{
	int** img_outa = (int**)IntAlloc2(height, width);
	int xdirect, ydirect;
	Xdirection(img, height, width, img_outa);
	Ydirection(img_outa, height, width, img_out);
	
}

*/

//1. 1차 미분연산
void MagGradient(int** img, int height, int width, int** img_out) 
{
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			int f_x = img[y][MIN(x + 1,width-1)] - img[y][x];
			int f_y = img[MIN(y+1,height-1)][x] - img[y][x];
			img_out[y][x] = abs(f_x) + abs(f_y);
		}
	}
}

void Normalize(int** img, int** img_out, int height, int width) {
	
	int maxvalue = FindMaxValue(img, height, width);
	//printf("\n max_value = %d", mvalue);

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			
		img_out[y][x] = (int)(255.0 * img[y][x] / maxvalue+0.5);
		}
	}
}
//	A붙은거랑 안붙은 거랑 똑같음 --> 파라미터만 다름

void NormalizeA(int** img, int height, int width) {

	int mvalue = FindMaxValue(img, height, width);
	//printf("\n max_value = %d", mvalue);

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {

			img[y][x] = (int)(255.0 * img[y][x] / mvalue + 0.5);
		}
	}
}
void main_EX1020() //1. 미분연산자 엣지검출 정석(평균필터 안하기 때문에 잡음에 민감)
{
	int height, width;
	int** img = ReadImage("lena.png", &height, &width); //img[y][x]
	int** img_out = (int**)IntAlloc2(height, width);
	int** img_out2 = (int**)IntAlloc2(height, width);

	MagGradient(img, height,  width,  img_out);
	
	Normalize(img_out, img_out2, height, width);

	ImageShow("입력영상", img, height, width);
	ImageShow("출력영상_정규화전", img_out, height, width);
	ImageShow("출력영상_정규화후", img_out2, height, width);
}

void AbsImage(int** img, int height, int width)
{
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			if (img[y][x] >= 0)
				img[y][x] = img[y][x];
			else if (img[y][x] < 0)
				img[y][x] = -img[y][x];
		}
	}
}

//라플라스 연산자 --> 마스크에서 계수만 바꿔주면됨

void main_EX1022()//엣지 검출 : MeanNxNFilterWithMask -> 절댓값 -> 정규화
{  //마스킹-->라플라스연산으로 발전시키기 (mask3이용)
	int height, width;
	int** img = ReadImage("lena.png", &height, &width); //img[y][x]

	int** img_out0 = (int**)IntAlloc2(height, width);
	int** img_out1 = (int**)IntAlloc2(height, width);
	int** img_out2 = (int**)IntAlloc2(height, width);
	int** img_out3 = (int**)IntAlloc2(height, width);
	int** img_out4 = (int**)IntAlloc2(height, width);
	int** img_out0_2 = (int**)IntAlloc2(height, width);
	int** img_out1_2 = (int**)IntAlloc2(height, width);
	int** img_out2_2 = (int**)IntAlloc2(height, width);
	int** img_out3_2 = (int**)IntAlloc2(height, width);
	int** img_out4_2 = (int**)IntAlloc2(height, width);

	float ** mask0 = (float**)FloatAlloc2(3, 3);
	float ** mask1 = (float**)FloatAlloc2(3, 3);
	float ** mask2 = (float**)FloatAlloc2(3, 3);
	float ** mask3 = (float**)FloatAlloc2(3, 3);

	mask0[0][0] = 1.0 / 9.0; mask0[0][1] = 1.0 / 9.0; mask0[0][2] = 1.0 / 9.0;
	mask0[1][0] = 1.0 / 9.0;  mask0[1][1] = 1.0 / 9.0; mask0[1][2] = 1.0 / 9.0;
	mask0[2][0] = 1.0 / 9.0;  mask0[2][1] = 1.0 / 9.0; mask0[2][2] = 1.0 / 9.0;

	//f'(x)=f(x-1)-f(x+1)  --> noise한 영상이 조금 줄음
	//세로엣지검출 
	mask1[0][0] = 1;  mask1[0][1] = 0; mask1[0][2] = -1;
	mask1[1][0] = 1; mask1[1][1] = 0; mask1[1][2] = -1;
	mask1[2][0] = 1; mask1[2][1] = 0; mask1[2][2] = -1;

	//f'(y)=f(y-1)-f(y+1)
	//가로엣지검출
	mask2[0][0] = 1; mask2[0][1] = 1; mask2[0][2] = 1;
	mask2[1][0] = 0; mask2[1][1] = 0; mask2[1][2] = 0;
	mask2[2][0] = -1;  mask2[2][1] = -1; mask2[2][2] = -1;

	//2. 라플라스를 이용한 엣지 검출
	mask3[0][0] = -1; mask3[0][1] = -1; mask3[0][2] = -1;
	mask3[1][0] = -1; mask3[1][1] = 8; mask3[1][2] = -1;
	mask3[2][0] = -1;  mask3[2][1] = -1; mask3[2][2] = -1;

	//평균필터 취해줌 --> 잡음에 덜 민감하게
	MeanNxNFilterWithMask(mask0, 3, img, height, width, img_out0);
	MeanNxNFilterWithMask(mask1, 3, img, height, width, img_out1);
	MeanNxNFilterWithMask(mask2, 3, img, height, width, img_out2);
	MeanNxNFilterWithMask(mask3, 3, img, height, width, img_out3);
	
	AbsImage(img_out0, height, width); 
	Normalize(img_out0, img_out0_2, height, width);
	AbsImage(img_out1, height, width); 
	Normalize(img_out1, img_out1_2, height, width);
	AbsImage(img_out2, height, width);
	Normalize(img_out2, img_out2_2, height, width);

	//라플라시안 연산자 : 방향성이 없음!
	AbsImage(img_out3, height, width);
	Normalize(img_out3, img_out3_2, height, width);

	// 1차 미분연산 평균필터X (지난 시간에 구현한것)
	MagGradient(img,height, width, img_out4);
	Normalize(img_out4, img_out4_2, height, width);

	//ImageShow("입력영상", img, height, width);
	ImageShow("출력영상0", img_out0_2, height, width);
	ImageShow("출력영상1", img_out1_2, height, width);
	ImageShow("출력영상2", img_out2_2, height, width);
	ImageShow("출력영상3", img_out3_2, height, width);
	ImageShow("출력영상4", img_out4_2, height, width);


}



void MagSobel_X(int** img, int height, int width, int** img_out) //소벨 수직
{
	float** mask_SOBEL = (float**)FloatAlloc2(3, 3); // float mask[3][3]; 와 다르다

	mask_SOBEL[0][0] = 1; mask_SOBEL[0][1] = 0; mask_SOBEL[0][2] = -1;
	mask_SOBEL[1][0] = 2; mask_SOBEL[1][1] = 0; mask_SOBEL[1][2] = -2;
	mask_SOBEL[2][0] = 1; mask_SOBEL[2][1] = 0; mask_SOBEL[2][2] = -1;

	MeanNxNFilterWithMask(mask_SOBEL, 3, img, height, width, img_out);

	AbsImage(img_out, height, width);

	FloatFree2(mask_SOBEL, 3, 3);

}



void MagSobel_Y(int** img, int height, int width, int** img_out) //소벨 수평
{
	float** mask_SOBEL = (float**)FloatAlloc2(3, 3); // float mask[3][3]; 와 다르다
	
	mask_SOBEL[0][0] = -1; mask_SOBEL[0][1] = -2; mask_SOBEL[0][2] = -1;
	mask_SOBEL[1][0] = 0; mask_SOBEL[1][1] = 0; mask_SOBEL[1][2] = 0;
	mask_SOBEL[2][0] = 1; mask_SOBEL[2][1] = 2; mask_SOBEL[2][2] = 1;
	/*
	mask_SOBEL[0][0] = 1; mask_SOBEL[0][1] = 2; mask_SOBEL[0][2] = 1;
	mask_SOBEL[1][0] = 0; mask_SOBEL[1][1] = 0; mask_SOBEL[1][2] = 0;
	mask_SOBEL[2][0] = -1; mask_SOBEL[2][1] = -2; mask_SOBEL[2][2] = -1;
	*/
	MeanNxNFilterWithMask(mask_SOBEL, 3, img, height, width, img_out);
	AbsImage(img_out, height, width);
	FloatFree2(mask_SOBEL, 3, 3);

}


void AddImages(int** img1, int** img2, int height, int width, int** img_out)
{
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			img_out[y][x] = img1[y][x] + img2[y][x];
		}
	}
}


void MagSobel(int** img, int height, int width, int** img_out)
{
	int** img_out_x = (int**)IntAlloc2(height, width);
	int** img_out_y = (int**)IntAlloc2(height, width);

	MagSobel_X(img, height, width, img_out_x);
	MagSobel_Y(img, height, width, img_out_y);

	AddImages(img_out_x, img_out_y, height, width, img_out);

	IntFree2(img_out_x, height, width);
	IntFree2(img_out_y, height, width);

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//수업시간에 한 sobel main 
//3. 소벨을 이용한 엣지검출(한번씩)
void main_sobeledge()// 1027과제 교수님 코드 
{

	int height, width;
	int** img = ReadImage("lena.png", &height, &width); // img[y][x]
	int** img_out0 = (int**)IntAlloc2(height, width);
	int** img_out1 = (int**)IntAlloc2(height, width);
	int** img_out1_2 = (int**)IntAlloc2(height, width);
	int** img_out2 = (int**)IntAlloc2(height, width);
	int** img_out2_2 = (int**)IntAlloc2(height, width);
	int** img_out3 = (int**)IntAlloc2(height, width);
	int** img_out3_2 = (int**)IntAlloc2(height, width);
	int** img_out4 = (int**)IntAlloc2(height, width);
	int** img_out4_2 = (int**)IntAlloc2(height, width);

	MagSobel(img, height, width, img_out1);
	MagSobel_X(img, height, width, img_out2);
	MagSobel_Y(img, height, width, img_out3);

	Normalize(img_out1, img_out1_2, height, width);
	Normalize(img_out2, img_out2_2, height, width );
	Normalize(img_out3, img_out3_2, height, width);

	ImageShow("입력영상", img, height, width);
	ImageShow("출력영상_x,y방향 합친거", img_out1_2, height, width);
	ImageShow("출력영상_x방향", img_out2_2, height, width);
	ImageShow("출력영상_y방향", img_out3_2, height, width);

}

// 1027과제 내 코드
void main_EX1027과제_sobel() // 3. 소벨을 이용한 엣지검출(2번씩 돌리기)
{   

	int height, width;
	int** img = (int**)ReadImage("lena.png", &height, &width); //img[y][x]
	int** img_out1 = (int**)IntAlloc2(height, width);
	int** img_out1_1 = (int**)IntAlloc2(height, width);
	int** img_out2 = (int**)IntAlloc2(height, width);
	int** img_out2_1 = (int**)IntAlloc2(height, width);
	int** img_out3 = (int**)IntAlloc2(height, width);
	int** img_out3_1 = (int**)IntAlloc2(height, width);

	float ** mask1 = (float**)FloatAlloc2(3, 3);
	float ** mask2 = (float**)FloatAlloc2(3, 3);

	
	mask1[0][0] = -1;  mask1[0][1] = -2; mask1[0][2] = -1;
	mask1[1][0] = 0; mask1[1][1] = 0; mask1[1][2] = 0;
	mask1[2][0] = 1; mask1[2][1] = 2; mask1[2][2] = 1;


	mask2[0][0] = 1; mask2[0][1] = 0; mask2[0][2] = -1;
	mask2[1][0] = 2; mask2[1][1] = 0; mask2[1][2] = -2;
	mask2[2][0] = 1;  mask2[2][1] = 0; mask2[2][2] = -1;

	MeanNxNFilterWithMask(mask1, 3, img, height, width, img_out1);
	MeanNxNFilterWithMask(mask2, 3, img, height, width, img_out2);
	
	
	MagSobel_X(img_out1, height, width, img_out1_1);
	NormalizeA(img_out1_1, height, width);

	MagSobel_Y(img_out2, height, width, img_out2_1);
	NormalizeA(img_out2_1, height, width);

	AddImages(img_out1_1, img_out2_1, height, width, img_out3_1);

	MagSobel(img_out3_1, height, width, img_out3_1);
	NormalizeA(img_out3_1, height, width);

	ImageShow("입력영상", img, height, width);
	ImageShow("출력영상_X", img_out1_1, height, width);
	ImageShow("출력영상_Y", img_out2_1, height, width);
	ImageShow("출력영상_X+Y", img_out3_1, height, width);
	
}


//10.29.(목) 수업내용
void Sharpening(float alpha, int** img, int height, int width, int** img_out) {

	//1. mask
	//2. filtering
	//3. clipping :if x<0 -> x=0, x>255 x=255
	//4. 이미지아웃
	float ** mask = (float**)FloatAlloc2(3, 3);
	float ** mask1 = (float**)FloatAlloc2(3, 3);

	// 더해서 1이 되야함
	mask[0][0] = 0; mask[0][1] = -alpha; mask[0][2] = 0;
	mask[1][0] = -alpha; mask[1][1] = 1 + 4 * alpha; mask[1][2] = -alpha;
	mask[2][0] = 0;  mask[2][1] = -alpha; mask[2][2] = 0;
	
	mask1[0][0] = -alpha; mask1[0][1] = -alpha; mask1[0][2] = -alpha;
	mask1[1][0] = -alpha; mask1[1][1] = 1+8*alpha ; mask1[1][2] = -alpha;
	mask1[2][0] = -alpha;  mask1[2][1] = -alpha; mask1[2][2] = -alpha;

	MeanNxNFilterWithMask(mask1, 3, img, height, width, img_out);
	
	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{

			img_out[y][x] = Clipping(img_out[y][x]); //해야함
		}
	}
	
}

// EX1029 선명화 처리
void main_EX1029() {
	int height, width;
	int** img = ReadImage("lena.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);
	float alpha = 1;
	Sharpening(alpha, img, height, width, img_out);

	ImageShow("입력영상", img, height, width);
	ImageShow("출력영상", img_out, height, width);
}

//배열에서 중간값 찾기
void example_of_sorting()
{
	float a[9] = { 1.0, 2.0, -1.0, -3.0, -4.0, 1.0, 1.2, 4.1, 0.1 };
	sort(a, a + 9); // 오름차순 
	for (int i = 0; i < 9; i++)
		printf("\n %f", a[i]);
	float med_value = a[4]; //4= (9-1) / 2
	printf("\n %f", med_value);

}

void main_sort() { 
	example_of_sorting();

}
/*
void MedianFiltering_MY(int** img, int height, int width, int** img_out) // 중간값 필터링 하기
{
	float b[9];
	int index = 0;
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {

			if (x < 1 || y < 1 || x >= width - 1 || y >= height - 1)
			{
				img_out[y][x] = img[y][x];
			}

			else
			{
				float b[9] = { img[y][x], img[y - 1][x + 1] ,img[y - 1][x - 1], img[y][x + 1] ,img[y][x - 1]
					,img[y + 1][x + 1] ,img[y + 1][x - 1] , img[y - 1][x],img[y + 1][x] };
				sort(b, b + 9);
				img_out[y][x] = b[4];
				
				
			}

		}

	}
}
*/


void MedianFiltering(int** img, int height, int width, int** img_out) // 중간값 필터링 하기
{
	
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			float b[9] = {};
			int index = 0;
			for (int dy = -1; dy <= 1; dy++) {
				for (int dx = -1; dx <= 1; dx++) {
					int new_y = _MIN_(_MAX_(y + dy, 0), height - 1);
					int new_x = _MIN_(_MAX_(x + dx, 0), width - 1);
					
					b[index] = img[new_y][new_x];
					index++;
				}
			}
			//sort(b[0], b[8]); --> 이렇게 sort함수안에 배열값 못 넣어 !!!! 계속 이상한 오류 뜨더라고 
			sort(b, b + 9); // *** 이렇게해
			img_out[y][x] = b[4];
		}
	}
}
void main_EX1103_과제제출() // 중간값필터링을 num 번 적용하는 함수 --> num 번 적용한 결과 한번만 띄움.
{
	int height, width;
	int** img = ReadImage("lenaSP20.png", &height, &width);
	int** img_out1 = (int**)IntAlloc2(height, width);
	int** img_out2 = (int**)IntAlloc2(height, width);
	int num = 10;// MedianFiltering 할 횟수
	CopyImage(img, height, width, img_out1);
	for (int i = 0; i < num - 1; i++) {
		MedianFiltering(img_out1, height, width, img_out2);
		CopyImage(img_out2, height, width, img_out1);
	}
	MedianFiltering(img_out1, height, width, img_out2);
	ImageShow("입력영상", img, height, width);
	ImageShow("출력영상", img_out2, height, width);
}

void main_EX1103_SOL() // 중간값필터링을 num번 적용하는 함수  //한 화면에 계속뜸
{
	int height, width;
	int** img = ReadImage("lenaSP10.png", &height, &width);
	int** img_out1 = (int**)IntAlloc2(height, width);
	int** img_out2 = (int**)IntAlloc2(height, width);
	ImageShow("입력영상", img, height, width);

	int num = 10;// MedianFiltering 할 횟수

	CopyImage(img, height, width, img_out1);
	
	for (int i = 0; i < num; i++) {
		MedianFiltering(img_out1, height, width, img_out2);
		CopyImage(img_out2, height, width, img_out1);
		ImageShow("출력영상", img_out2, height, width); 
	}
	
}


// *******두배 확대 과제*******
void UpscaleX2(int** img, int height, int width, int** img_up)
{
	int height2 = height * 2;
	int width2 = width * 2;
	

	for (int y = 0; y < height; y ++) {
		for (int x = 0; x < width; x ++) {
			img_up[2*y][2*x] = img[y][x];
		}
	}
	for (int y = 0; y <height2; y += 2) {
		for (int x = 1; x <width2; x += 2) {
			img_up[y][x] = (int)((img_up[y][x - 1] + img_up[y][_MIN_(x + 1, width2 - 2)]) / 2.0+0.5);
		}
	}
	for (int y = 1; y < height2; y += 2) {
		for (int x = 0; x <  width2; x++) {
			img_up[y][x] = (int)((img_up[y - 1][x] + img_up[_MIN_(y + 1, height2 - 2)][x]) / 2.0+0.5);
		}
	}
	// 예외처리가 필요하면 하기

	/*
	int x = width2 - 1;
	for (int y = 0; y < height2; y++)
	{
		img_up[y][x] = img_up[y][x - 1];
	}
	int y = height2 - 1;
	for (int x = 0; x < width2; x++)
	{
		img_up[y][x] = img_up[y-1][x];
	}
	*/
	
}
//2배 3배 확대?
void UpscaleX2_2(int** img, int height, int width, int** img_up)
{
	int height2 = height * 2;
	int width2 = width * 2;


	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			img_up[2 * y][2 * x] = img[y][x];
		}
	}
	for (int y = 0; y <height2; y += 2) {
		for (int x = 1; x <width2; x += 2) {
			if (x == width2 - 1) {
				img_up[y][x] = img_up[y][x - 1];
			}
			else
				img_up[y][x] = (int)((img_up[y][x - 1] + img_up[y][x + 1]) / 2.0 + 0.5);
		}
	}
	for (int y = 1; y < height2; y += 2) {
		for (int x = 0; x < width2; x++) {
			if (y == height2 - 1) {
				img_up[y][x] = img_up[y-1][x ];
			}
			else
				img_up[y][x] = (int)((img_up[y-1][x] + img_up[y+1][x ]) / 2.0 + 0.5);
			
		}
	}
	

}

void DownscaleX2(int** img, int height,int width, int** img_up)
{
	


	for (int y = 0; y < height; y+=2) {
		for (int x = 0; x < width; x+=2) {
			img_up[y/2][x/2] = img[y][x];
		}
	}


}

void main_EX1105() // 2배 확대 및 축소
{
	int height, width;
	int** img = ReadImage("lena.png", &height, &width);
	int** img_up = (int**)IntAlloc2(height*2, width*2);
	int** img_down = (int**)IntAlloc2(height/2, width/2);
	UpscaleX2_2(img, height, width,img_up);
	DownscaleX2(img, height, width, img_down);
	
	ImageShow("입력영상", img, height, width);
	ImageShow("두배확대 출력영상", img_up, height*2, width*2);
	ImageShow("두배축소 출력영상", img_down, height/2, width/2);
}

///////////////////////////////////////// 11/12 ////////////////////////////////////////
bool Is_Inside(int y, int x, int height, int width) {
	if (y < 0 || y >= height) return(false);
	if (x < 0 || x >= width) return(false);

	return(true);
}

//여기서 X Y 는 픽셀의 회전된 위치
//보간법
int BilinearInterpolation(float y, float x, int**img, int height, int width ) {
	int output_x;
	
	int a_y = (int)y;
	int a_x = (int)x;
	int b_y = a_y;
	int b_x = a_x +1;
	int c_y = a_y + 1;
	int c_x = a_x;
	int d_y = a_y + 1;
	int d_x = a_x + 1;
	
	// 넘어가는 부분을 0으로 채우는 작업
	if (Is_Inside(a_y, a_x, height, width)==false) return(0);
	if (Is_Inside(b_y, b_x, height, width) == false) return(0);
	if (Is_Inside(c_y, c_x, height, width) == false) return(0);
	if (Is_Inside(d_y, d_x, height, width) == false) return(0);

	int a = img[a_y][a_x];
	int b = img[b_y][b_x];
	int c = img[c_y][c_x];
	int d = img[d_y][d_x];

	float delta_y = y - a_y;
	float delta_x = x - a_x;
	
	output_x = (int)((1 - delta_y)*(1 - delta_x)*a + delta_x*(1 - delta_y)*b
		+ (1 - delta_x)*delta_y*c + delta_x*delta_y*d + 0.5);

	return output_x;

}
// 왼쪽 위를 기준으로 회전
void RotateImage(float theta, int** img, int height, int width, int** img_out)
{

	for (int y_prime = 0; y_prime < height; y_prime++) {
		for (int x_prime = 0; x_prime < width; x_prime++) {
			float x = cos(theta) * (x_prime) + sin(theta) * (y_prime);
			float y = sin(theta) * (-x_prime) + cos(theta) * (y_prime);
			img_out[y_prime][x_prime] = BilinearInterpolation(y, x, img, height, width);
		}
	}
}
//시계 반대방향 회전 아직 미완
void RotateImage_1(float theta, int** img, int height, int width, int** img_out)
{

	for (int y_prime = height-1 ; y_prime >=0; y_prime--) {
		for (int x_prime = width-1; x_prime >= 0; x_prime--) {
			float x = cos(theta) * (x_prime)+sin(theta) * (y_prime);
			float y = sin(theta) * (-x_prime) + cos(theta) * (y_prime);
			img_out[y_prime][x_prime] = BilinearInterpolation(y, x, img, height, width);
		}
	}
}


// 중심에서 회전

void RotateAndScaleImage(float a, float theta, int** img, int height, int width, int** img_out)
{
	float xcenter = (double)width / 2.0;
	float ycenter = (double)height / 2.0;  // 중심을 바꾸려면 여길 바꾸면됨
	float cc = cos(theta);
	float ss = sin(theta);
	float b = 1 / a;

	for (int y_prime = 0; y_prime < height; y_prime++) {
		for (int x_prime = 0; x_prime < width; x_prime++) {
			float x = (xcenter + b*(y_prime - ycenter)* ss + b*(x_prime - xcenter)*cc) ;
			float y =(ycenter + b*(y_prime - ycenter)* cc - b*(x_prime - xcenter)*ss);

			/* 반시계방향 방법1
			float x = (xcenter + b*(y_prime - ycenter)* ss - b*(x_prime - xcenter)*cc) ;
			float y =(ycenter + b*(y_prime - ycenter)* cc + b*(x_prime - xcenter)*ss);
			*/

			/* 특정 위치 x0,y0를 중심으로 회전 
			x2 = ( y1 - y0 ) * sin(seta) + ( x1 - x0 ) * cos(seta) + x0;
			y2 = ( y1 - y0 ) * cos(seta) - ( x1 - x0 ) * sin(seta) + y0;
			*/

			img_out[y_prime][x_prime] = BilinearInterpolation(y, x, img, height, width);
		}
	}
}

// 이미지 원점 기준으로 a 배만큼 이미지 확대 축소 (같은 512*512 프레임 안에서)

void ScaleImage(float a, int** img, int height, int width, int** img_out) {
	float b = 1 / a;
	int p = width;
	int t = height;
	for (int y_prime = 0; y_prime < height; y_prime++) {
		for (int x_prime = 0; x_prime < width; x_prime++) {
			float x = b* (x_prime);
			float y = b*  (y_prime) ;

			
					img_out[y_prime][x_prime] = BilinearInterpolation(y, x, img, height, width);
				}
		}
}
	


void main_EX1112() //Affme Transform 메인 
{
	int height, width;
	int** img = ReadImage("lena.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width );

	float theta = (CV_PI / 4.0);
	//CV_PI*2 -(CV_PI / 4.0)  // 반시계방향 방법2
	float a = 0.5; // 회전하면서 축소를 얼마나 할건지  //a가 작을 수록 이미지 작아짐

	//RotateImage(theta, img, height, width, img_out);
	//RotateAndScaleImage(a, theta, img, height, width, img_out);
	ScaleImage(a,img, height, width, img_out);
	ImageShow("입력영상", img, height, width);

	ImageShow("출력영상", img_out, height, width);
}

//MAD 구하기 
float MAD(int y_A, int x_A, int**img, int height, int width, int**tpl, int h_tpl, int w_tpl)
{
	float mad=0.0;
	//int sum = 0;
			for (int y = 0; y < h_tpl; y++) {
				for (int x = 0; x < w_tpl; x++) {
			
							
							mad += (float)abs(img[y+y_A][x+x_A] - tpl[y][x]);
							
				
					
				}
			}
			mad = mad * (1.0 / (h_tpl*w_tpl));
			return mad;
		
}

//MAD의 최소값 구하기 
float TemplateMatching(int height, int width, int h_tpl, int w_tpl, int**img, int**tpl , int* y_min_out, int* x_min_out) {

	float mad_min = INT_MAX; //초기치  OR 255
	int y_min, x_min;

	for (int y_A = 0; y_A <= height - h_tpl; y_A++) {
		for (int x_A = 0; x_A <= width - w_tpl; x_A++) {
			float mad = MAD(y_A, x_A, img, height, width, tpl, h_tpl, w_tpl);
			if (mad < mad_min) {

				mad_min = mad;
				y_min = y_A;
				x_min = x_A;
			}
		}
	}
	printf("MAD_min = %f , y_min = %d, x_min = %d", mad_min, y_min, x_min);
	*y_min_out = y_min;
	*x_min_out = x_min;

	return mad_min;//return에서 프로그램 끝남
}

//템플릿 매칭 
void main_EX1118()
{
	int height, width;
	int** img = ReadImage("lena.png", &height, &width);
	int h_tpl, w_tpl;
	int** tpl= ReadImage("lena_template.png", &h_tpl, &w_tpl);
	int y_min = 0, x_min = 0;

	float mad_min = TemplateMatching(height,  width, h_tpl, w_tpl,img, tpl, &y_min, &x_min);
	printf("MAD_min = %f , y_min = %d, x_min = %d \n", mad_min, y_min, x_min);
	
	for (int y = 0; y < h_tpl; y++) {
		for (int x = 0; x < w_tpl; x++) {
			printf("%d", img[y_min + y][x_min + x]);
		}
	}
}






void main_EX1124()
{
	int a = 10;
	int* c; 
	c = &a;

	printf("a = %d, *c = %d, c=%0x", a, *c, c);
}
void fun(int b[5]) { // = int* b
	for (int i = 0; i < 5; i++)
		printf("%d, %d \n", i, b[i]);
}
void fun1(int* b) { // = int* b
	
		printf("%d \n", *(b+1));
}
void main_EX1124_1()
{
	int a[5] = { 10,20,30,40,50 };
	fun1(a);


}

//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////////////

//11월 25일  흑백 모자익 --> 미완성
float MAD_mo1(int y_A, int x_A, int**img, int height, int width, int**tpl, int h_tpl, int w_tpl)
{
	float mad[510] = { 0.0 };
	//int sum = 0;
	for (int i = 0; i < 510; i++) {
		for (int y = 0; y < h_tpl; y++) {
			for (int x = 0; x < w_tpl; x++) {


				mad[i] += (float)abs(img[y + y_A][x + x_A] - tpl[y][x]);



			}
		}
		mad[i] = mad[i] * (1.0 / (h_tpl*w_tpl));
		
	}
	float mad_min = mad[0];
	int mad_min_index = 0;

	for (int a = 1; a < 500; a++) {
		if (mad[a] < mad_min)
		{
			mad_min = mad[a];
			mad_min_index = a;
		}

	}
	return mad_min_index;

}

#define DB_NUM 510


void main_EX1125()// 흑백 모자익 --> 미완성 
{


	char db_dir[300] = "C:\\Users\\user\\Desktop\\김민정\\2020-2수업\\지능형영상처리\\솔루션\\지능형영상처리day1\\db_students";
	char db_fname_prefix[100] = "dbs";
	char db_fname[300];

	int db_height, db_width;
	int** db_imgs[DB_NUM];



	for (int i = 0; i < DB_NUM; i++) {

		sprintf(db_fname, "%s\\%s%04d.jpg", db_dir, db_fname_prefix, i);
		db_imgs[i] = ReadImage(db_fname, &db_height, &db_width);
		//printf("\n[%d] (%d, %d)", i, db_height, db_width);
	}
	int height, width;
	int** img = ReadImage("lena.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);


	int index;
	float mad[DB_NUM] = { 0.0 };
	//int sum = 0;


	for (int y_A = 0; y_A <= height - db_height; y_A += 32) {
		for (int x_A = 0; x_A <= width - db_width; x_A += 32) {
			
			index = MAD_mo1(y_A, x_A, img, height, width, db_imgs[DB_NUM], db_height, db_width);
			for (int y = 0; y < db_height; y++) {
				for (int x = 0; x < db_width; x++) {
					img[y_A + y][x_A + x] = img[y_A + y][x_A + x] + db_imgs[index][y][x];
				}
			}

		}
	
	
	}
	ImageShow("입력영상", img, height, width);
	ImageShow("출력영상", img_out, height, width);
}




///////////////////////사이즈 조정/////////////////////
//8*8로 바꿔서 해봅시다.
//픽셀4개를 하나로 줄여서 하기

void main_color1() //칼라 영상 읽어들이기+출력하기
{

	int height, width;
	int_rgb** img = ReadColorImage("min.png",  &height,  &width);
	//int** img_out = (int**)IntAlloc2(height, width);

	int x0 = width / 4;
	int y0 = height / 4;
	int w_box = width / 2;
	int h_box = height / 2;

	for (int y = 0; y < h_box; y++) {
		for (int x = 0; x < w_box; x++) {
			img[y + y0][x + x0].b = 255;
			img[y + y0][x + x0].g = 255;
			img[y + y0][x + x0].r = 255;
		}
	}
	
	ColorImageShow("입력영상", img, height, width);
	//ImageShow("출력영상", img_out, height, width);
}

void main_color_read_print() //칼라 영상 읽어들이기+출력하기
{

	char db_dir[300] = "C:\\Users\\user\\Desktop\\김민정\\2020-2수업\\지능형영상처리\\솔루션\\지능형영상처리day1\\db_students";
	char db_fname_prefix[100] = "dbs";
	char db_fname[300];
	int db_height, db_width;


	int_rgb** db_imgs[DB_NUM];
	
	for (int i = 0; i < DB_NUM; i++) {
		sprintf(db_fname, "%s\\%s%04d.jpg", db_dir, db_fname_prefix, i);
		db_imgs[i] = ReadColorImage(db_fname, &db_height, &db_width);
	}
	
		for (int n = 1; n < DB_NUM; n++) {
			float mad = 0.0;
			for (int y = 0; y < db_height; y++) {
				for (int x = 0; x < db_width; x++) {
					mad += abs(db_imgs[0][y][x].b - db_imgs[n][y][x].b) +
						abs(db_imgs[0][y][x].g - db_imgs[n][y][x].g) +
						abs(db_imgs[0][y][x].r - db_imgs[n][y][x].r);
				}
			}
			mad = mad / (db_height*db_width * 3);
			printf("\n[%d] mad = %.2f", n, mad);
		}

		

	

}
void MakeHalf_1(int_rgb*** db_imgs32, int num, int db_height, int db_width, int_rgb*** db_imgs16) // 컬러 사이즈 반으로 줄이는 함수 
{
	for (int k = 0; k < num; k++)
		for (int y = 0; y < db_height; y += 2) {
			for (int x = 0; x < db_width; x += 2) {
				(db_imgs16[0][y / 2][x / 2].r = (db_imgs32[0][y][x]).r
					+ db_imgs32[0][y][x + 1].r
					+ (db_imgs32[0][y + 1][x]).r
					+ (db_imgs32[0][y + 1][x + 1]).r + 2) / 4;

				(db_imgs16[0][y / 2][x / 2].g = (db_imgs32[0][y][x]).g
					+ db_imgs32[0][y][x + 1].g
					+ (db_imgs32[0][y + 1][x]).g
					+ (db_imgs32[0][y + 1][x + 1]).g + 2) / 4;

				(db_imgs16[0][y / 2][x / 2].b = (db_imgs32[0][y][x]).b
					+ db_imgs32[0][y][x + 1].b
					+ (db_imgs32[0][y + 1][x]).b
					+ (db_imgs32[0][y + 1][x + 1]).b + 2) / 4;
			}
		}
}
void main_color_size_change() // 칼라 DB이미지들 사이즈 조정. 
{

	char db_dir[300] = "C:\\Users\\user\\Desktop\\김민정\\2020-2수업\\지능형영상처리\\솔루션\\지능형영상처리day1\\db_students";
	char db_fname_prefix[100] = "dbs";
	char db_fname[300];
	int db_height, db_width;


	int_rgb** db_imgs32[DB_NUM];

	for (int i = 0; i < DB_NUM; i++) {
		sprintf(db_fname, "%s\\%s%04d.jpg", db_dir, db_fname_prefix, i);
		db_imgs32[i] = ReadColorImage(db_fname, &db_height, &db_width);
	}

	int_rgb** db_imgs16[DB_NUM];

	for (int i = 0; i < DB_NUM; i++) {
		
		db_imgs16[i] = (int_rgb**)IntColorAlloc2(db_height/2, db_width/2);
	}
	
	MakeHalf_1(db_imgs32, DB_NUM, db_height, db_width, db_imgs16);


	int_rgb** db_imgs8[DB_NUM];

	for (int i = 0; i < DB_NUM; i++) {

		db_imgs8[i] = (int_rgb**)IntColorAlloc2(db_height /4, db_width / 4);
	}

	MakeHalf_1(db_imgs16, DB_NUM, db_height, db_width, db_imgs8);
	


}

void MakeHalf_glay(int*** db_imgs32, int num, int db_height, int db_width, int*** db_imgs16)
{
	for (int k = 0; k < num; k++)
		for (int y = 0; y < db_height; y += 2) {
			for (int x = 0; x < db_width; x += 2) {
				db_imgs16[k][y / 2][x / 2] = ((db_imgs32[k][y][x]) + db_imgs32[k][y][x + 1] + (db_imgs32[k][y + 1][x]) + (db_imgs32[k][y + 1][x + 1]) + 2) / 4;
			}
		}
}
void FIRST_Mosaic() {
#define DB_NUM 510                                       
	char db_dir[300] = "C:\\Users\\yooji\\Desktop\\data\\db_students";  
	char db_fname_prefix[100] = "dbs";
	char db_fname[300];                                      
	int mosac_size;                                          

	int db_height, db_width;                                 
	int** db_imgs[DB_NUM];                                    
	int db_rgb_average[DB_NUM];                               

	int** db_imgs32[DB_NUM];

	for (int i = 0; i < DB_NUM; i++) {
		sprintf(db_fname, "%s\\%s%04d.jpg", db_dir, db_fname_prefix, i);
		db_imgs32[i] = ReadImage(db_fname, &db_height, &db_width);
	}


	for (int i = 0; i < DB_NUM; i++) {

		db_imgs[i] = (int**)IntColorAlloc2(db_height / 2, db_width / 2);
	}

	MakeHalf_glay(db_imgs32, DB_NUM, db_height, db_width, db_imgs);

	db_height /= 2;
	db_width /= 2;


	for (int i = 0; i < DB_NUM; i++) {
		
		int db_gray_sum = 0;

		mosac_size = db_height;
		for (int y = 0; y < db_height; y++) {
			for (int x = 0; x < db_width; x++) {
				db_gray_sum += db_imgs[i][y][x];

			}
		}
		db_rgb_average[i] = (db_gray_sum / (db_height * db_width));
	}

	int height, width;
	char original_img_name[100] = "lena.png";

	int** original_img = ReadImage(original_img_name, &height, &width);
	int** img_out = (int**)IntColorAlloc2(height, width);


	int gray_sum;
	int gray_average;

	for (int i = mosac_size; i <= height; i += mosac_size) {
		for (int j = mosac_size; j <= width; j += mosac_size) {
			gray_sum = 0;

			for (int y = i - mosac_size; y < i; y++) {
				for (int x = j - mosac_size; x < j; x++) {
					gray_sum += original_img[y][x];
				}
			}
			gray_average = (gray_sum / (mosac_size * mosac_size));


			int match_index = -1;
			int diff_value = -1;

			for (int i = 0; i < DB_NUM; i++) {
				int diff_gray;
				diff_gray = abs(db_rgb_average[i] - gray_average);



				if (diff_value == -1) {
					diff_value = diff_gray;
					match_index = i;
				}
				else {
					if (diff_value > diff_gray) {
						diff_value = diff_gray;
						match_index = i;
					}
				}
			}

			int db_y = 0;

			for (int y = i - mosac_size; y < i; y++) {
				int db_x = 0;
				for (int x = j - mosac_size; x < j; x++) {
					
					img_out[y][x] = db_imgs[match_index][db_y][db_x];
					db_x++;
				}
				db_y++;
			}
		}
	}


	ImageShow("result", img_out, height, width);



}


void main_1212()
{
	FIRST_Mosaic();
	
}


void WriteConstBlock(int_rgb** img, int brightness, int pos_y, int pos_x, int bl_size, int** img_out) { //랜덤한 위치에 랜덤한 색상 넣기



	for (int y = 0; y < bl_size; y++) {
		for (int x = 0; x < bl_size; x++) {
			img_out[y + pos_y][x + pos_x] =  brightness;
		}
	}

}
void MakeRandomBoxImage(int num_boxes, int** img_out, int height, int width) { //랜덤하게 밝기, 사이즈, 위치 조정하기

	//int height, width;
	int_rgb** img = ReadColorImage("min.png", &height, &width);

	int table[4] = { 4,8,16,32 };
	srand((unsigned int)time(NULL));


	for (int k = 0; k < num_boxes; k++) {
		int brightness = rand() % 256;
		int bl_size = table[rand() % 4];
		int pos_y = rand() % (height - bl_size);
		int pos_x = rand() % (width - bl_size);

		WriteConstBlock(img, brightness, pos_y, pos_x, bl_size, img_out);
		
	}

}


void main_1() { // 블록 위치 불규칙하게 -> 배경이미지에서 (크기와) 위치가 랜덤하게
			 
	int height, width;
	int** img = ReadImage("lena.png", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);
	for (int i = 0; i < 10; i++) {
		MakeRandomBoxImage(300, img_out, height, width);
		
	}
	
		ImageShow("out", img_out, height, width);
	
}

void main_ex_db_nocolor_print() { 

	int height = 32;
	int  width = 32;
	int** img = ReadImage("dbs0002.jpg", &height, &width);


	ImageShow("out", img, height, width);


}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
//********************************************************텀프로젝트**************************************************************
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#define DB_NUM 510 

// db이미 반으로 줄이기
void MakeHalf(int_rgb*** db_imgs32, int num, int db_height, int db_width, int_rgb*** db_imgs16)
{
	for (int k = 0; k < num; k++)
		for (int y = 0; y < db_height; y += 2) {
			for (int x = 0; x < db_width; x += 2) {
				db_imgs16[k][y / 2][x / 2].r = ((db_imgs32[k][y][x]).r + db_imgs32[k][y][x + 1].r + (db_imgs32[k][y + 1][x]).r + (db_imgs32[0][y + 1][x + 1]).r + 2) / 4;

				db_imgs16[k][y / 2][x / 2].g = ((db_imgs32[k][y][x]).g
					+ db_imgs32[k][y][x + 1].g
					+ (db_imgs32[k][y + 1][x]).g
					+ (db_imgs32[k][y + 1][x + 1]).g + 2) / 4;

				db_imgs16[k][y / 2][x / 2].b = ((db_imgs32[k][y][x]).b
					+ db_imgs32[k][y][x + 1].b
					+ (db_imgs32[k][y + 1][x]).b
					+ (db_imgs32[k][y + 1][x + 1]).b + 2) / 4;
			}
		}
}


// db이미지들에 대해 r,g,b 각 값의 평균 구하기
// db_rgb_average배열의 i번쨰 인덱스의 r/g/b 에 db_imgs영상의 R/G/B 값의 평균을 삽입
void average_rgb(int_rgb*** db_imgs, int db_height, int db_width, int_rgb db_rgb_average[DB_NUM]) {
	for (int i = 0; i < DB_NUM; i++) {
		int db_r_sum = 0;
		int db_g_sum = 0;
		int db_b_sum = 0;
		for (int y = 0; y < db_height; y++) {
			for (int x = 0; x < db_width; x++) {
				db_r_sum += db_imgs[i][y][x].r;
				db_g_sum += db_imgs[i][y][x].g;
				db_b_sum += db_imgs[i][y][x].b;
			}
		}
		db_rgb_average[i].r = (db_r_sum / (db_height * db_width));
		db_rgb_average[i].g = (db_g_sum / (db_height * db_width));
		db_rgb_average[i].b = (db_b_sum / (db_height * db_width));

	}
}

// 배경영상의 모자이크 크기내 범의의 R값 평균값 구하기
int R_Average(int a, int b, int size, int_rgb**  original_img) {
	int r_sum = 0;
	int r_average = 0;
	for (int y = a - size; y < a; y++) {
		for (int x = b - size; x < b; x++) {
			r_sum += original_img[y][x].r;
		}
	}
	r_average = (r_sum / (size * size));
	return r_average;
}

// 배경영상의 모자이크 크기내 범의의 G값 평균값 구하기
int G_Average(int a, int b, int size, int_rgb**  original_img) {
	int g_sum = 0;
	int g_average = 0;
	for (int y = a - size; y < a; y++) {
		for (int x = b - size; x < b; x++) {
			g_sum += original_img[y][x].g;
		}
	}
	g_average = (g_sum / (size * size));
	return g_average;
}

// 배경영상의 모자이크 크기내 범의의 B값 평균값 구하기
int B_Average(int a, int b, int size, int_rgb**  original_img) {
	int b_sum = 0;
	int b_average = 0;

	for (int y = a - size; y < a; y++) {
		for (int x = b - size; x < b; x++) {

			b_sum += original_img[y][x].b;
		}
	}
	b_average = (b_sum / (size * size));
	return b_average;
}

// 배경영상의 모자이크 크기내 범위의 r,g,b값과 가장 비슷한 db이미지 = Match
// db[i]영상의 r/g/b 값의 평균- 배경영상의 모자이크 범위에 해당하는 평균 r/g/b값 ->  절대값 취하기 
int  Find_Match_Index(int_rgb db_rgb_average[DB_NUM], int r_average, int g_average, int b_average) {
	int match_index = -1;													// index를 비교하기위해 -1 값으로 초기화
	int difference = -1;													// 차이값을 비교하기위해 -1 값으로 초기화
	for (int a = 0; a < DB_NUM; a++) {
		int r_difference, g_difference, b_difference;
		r_difference = abs(db_rgb_average[a].r - r_average);
		g_difference = abs(db_rgb_average[a].g - g_average);
		b_difference = abs(db_rgb_average[a].b - b_average);
		if (difference == -1) {												//처음 구할땐 다음 과정 실행											
			match_index = a;
			difference = r_difference + g_difference + b_difference;
			match_index = a;
		}
		else {																//이후부턴 다음 과정 실행											
			if (difference > (r_difference + g_difference + b_difference)) {
				difference = r_difference + g_difference + b_difference;
				match_index = a;
			}
		}
	}
	return match_index;														//가장 비슷한 영상의 INDEX값을 return 하기.
}

// 출력할 img_out 영상 만들기
void OUT_IMAGE(int  MosaicSize, int height, int width, int_rgb db_rgb_average[DB_NUM], int_rgb** db_imgs[DB_NUM], int_rgb** original_img, int_rgb** img_out) {
	for (int i = MosaicSize; i <= height; i += MosaicSize) {										// 원하는 사이즈 간격마다 반복(16*16 /8*8 / ....마다 다름)
		for (int j = MosaicSize; j <= width; j += MosaicSize) {
			int r_sum = 0;																			//색상별 합 변수 선언&초기화
			int g_sum = 0;
			int b_sum = 0;
			int r_average = R_Average(i, j, MosaicSize, original_img);
			int g_average = G_Average(i, j, MosaicSize, original_img);
			int b_average = B_Average(i, j, MosaicSize, original_img);
			int match_index = Find_Match_Index(db_rgb_average, r_average, g_average, b_average);
			int db_img_y = 0;
			for (int y = i - MosaicSize; y < i; y++) {
				int db_img_x = 0;
				for (int x = j - MosaicSize; x < j; x++) {
					img_out[y][x] = db_imgs[match_index][db_img_y][db_img_x];						// 반복문을 돌리며 img_out[y][x]의 각 값에 db_imgs[match_index][db_y][db_x] 픽셀의 값을 넣어주기
					db_img_x++;
				}
				db_img_y++;
			}
		}
	}
}

// 랜덤하게 출력할 img_out 영상 만들기
void OUT_IMAGE_RANDOM(int  MosaicSize, int height, int width, int_rgb db_rgb_average[DB_NUM], int_rgb** db_imgs[DB_NUM], int_rgb** original_img, int_rgb** img_out) {

	for (int t = 0; t < 480; t++) {
		int pos_y = rand() % (height - MosaicSize);
		int pos_x = rand() % (width - MosaicSize);
		for (int i = pos_y + MosaicSize; i <= height; i += pos_y) {										    // 원하는 사이즈 간격마다 반복(16*16 /8*8 / ....마다 다름)
			for (int j = pos_x + MosaicSize; j <= width; j += pos_x) {
				int r_sum = 0;																			//색상별 합 변수 선언&초기화
				int g_sum = 0;
				int b_sum = 0;
				int r_average = R_Average(i, j, MosaicSize, original_img);
				int g_average = G_Average(i, j, MosaicSize, original_img);
				int b_average = B_Average(i, j, MosaicSize, original_img);
				int match_index = Find_Match_Index(db_rgb_average, r_average, g_average, b_average);
				int db_img_y = 0;
				for (int y = i - MosaicSize; y < i; y++) {
					int db_img_x = 0;
					for (int x = j - MosaicSize; x < j; x++) {
						img_out[y][x] = db_imgs[match_index][db_img_y][db_img_x];						// 반복문을 돌리며 img_out[y][x]의 각 값에 db_imgs[match_index][db_y][db_x] 픽셀의 값을 넣어주기
						db_img_x++;
					}
					db_img_y++;
				}
			}
		}
	}
}

//사이즈별로 모자이크하기
void set_img(int_rgb** db_imgs16[DB_NUM], int db_height16, int db_width16, int_rgb db_rgb_average[DB_NUM], int height, int width, int_rgb** original_img, int_rgb** img_out)
{
	average_rgb(db_imgs16, db_height16, db_width16, db_rgb_average);												//rgv 색상평균 구하기
	OUT_IMAGE(db_height16, height, width, db_rgb_average, db_imgs16, original_img, img_out);						//img_out 구하기
	if (db_height16> 10) ColorImageShow("모자이크16*16", img_out, height, width);									//해당 db이미지 사이즈에 맞는 이름으로 모자이크 영상 출력하기 
	else if (db_height16> 5) ColorImageShow("모자이크8*8", img_out, height, width);
	else  ColorImageShow("모자이크4*4", img_out, height, width);
}

//랜덤한 위치로 모자이크 하기
void set_img_RANDOM(int_rgb** db_imgs16[DB_NUM], int db_height16, int db_width16, int_rgb db_rgb_average[DB_NUM], int height, int width, int_rgb** original_img, int_rgb** img_out)
{
	average_rgb(db_imgs16, db_height16, db_width16, db_rgb_average);												//rgv 색상평균 구하기
	OUT_IMAGE_RANDOM(db_height16, height, width, db_rgb_average, db_imgs16, original_img, img_out);						//랜덤한 img_out 구하기
    ColorImageShow("랜덤 모자이크16*16", img_out, height, width);								
	
}

//16*16사이즈 db로 모자이크하기
void Mosaic16X16() {
	char db_dir[300] = "C:\\Users\\user\\Desktop\\김민정\\2020-2수업\\지능형영상처리\\솔루션\\지능형영상처리day1\\db_students";
	char db_fname_prefix[100] = "dbs";
	char db_fname[300];
	int db_height, db_width, db_height16, db_width16;																// db이미지의 높이, 너비 변수선언
	int_rgb** db_imgs32[DB_NUM];
	int_rgb** db_imgs16[DB_NUM];
	int_rgb db_rgb_average[DB_NUM];																					// db 이미지들의 평균 RGB를 담을 구조체형 배열
	int height, width;																								// 배경 영상의 높이와 너비 변수
	char original_img_name[100] = "스파이더맨.jpg";																	// 배경 영상의 파일 이름 ""안에 수정OK
	int_rgb** original_img = ReadColorImage(original_img_name, &height, &width);									// 배경영상(=원본)을 구조체로 읽어와 선언및 삽입
	if (height % 16 != 0) height -= height % 16;
	if (width % 16 != 0) width -= width % 16;
	int_rgb** img_out = (int_rgb**)IntColorAlloc2(height, width);													// 결과 영상 img_out	
	for (int i = 0; i < DB_NUM; i++) {
		sprintf(db_fname, "%s\\%s%04d.jpg", db_dir, db_fname_prefix, i);
		db_imgs32[i] = ReadColorImage(db_fname, &db_height, &db_width);
	}
	for (int i = 0; i < DB_NUM; i++)	db_imgs16[i] = (int_rgb**)IntColorAlloc2(db_height / 2, db_width / 2);		// 16*16 영상을 저장할 구조체의 메모리 할당
	MakeHalf(db_imgs32, DB_NUM, db_height, db_width, db_imgs16);													// db_imgs32 의 4픽셀을 하나의 픽셀로 합쳐 크기 줄인후  db_imgs16 배열에 저장
	db_height16 = db_height / 2;
	db_width16 = db_width / 2;
	set_img(db_imgs16, db_height16, db_width16, db_rgb_average, height, width, original_img, img_out);			//set_img() 함수 호출로 이미지  출력 
}



//8*8사이즈 db로 모자이크하기
void Mosaic8X8() {
	char db_dir[300] = "C:\\Users\\user\\Desktop\\김민정\\2020-2수업\\지능형영상처리\\솔루션\\지능형영상처리day1\\db_students";
	char db_fname_prefix[100] = "dbs";
	char db_fname[300];
	int db_height, db_width, db_height16, db_width16, db_height8, db_width8;							// 추가로  db_height8, db_width48선언
	int_rgb** db_imgs32[DB_NUM];
	int_rgb** db_imgs16[DB_NUM];
	int_rgb** db_imgs8[DB_NUM];																			// 추가로 선언
	int_rgb db_rgb_average[DB_NUM];																		// 각 db 이미지의 평균 RGB를 담을 int_rgb 형 구조체 배열;
	int height, width;																					// 배경영상의 높이와 너비 변수
	char original_img_name[100] = "스파이더맨.jpg";														// 배경 영상의 파일 이름
	int_rgb** original_img = ReadColorImage(original_img_name, &height, &width);						// 배경 영상(=원본)을 구조체로 읽어와 선언및 삽입
	if (height % 16 != 0) height -= height % 16;
	if (width % 16 != 0) width -= width % 16;
	int_rgb** img_out = (int_rgb**)IntColorAlloc2(height, width);										// 결과 값을 저장할 구조체의 메모리 할당
	for (int i = 0; i < DB_NUM; i++) {																	// db의 영상들 모두 읽고 배열에 저장
		sprintf(db_fname, "%s\\%s%04d.jpg", db_dir, db_fname_prefix, i);
		db_imgs32[i] = ReadColorImage(db_fname, &db_height, &db_width);
	}
	for (int i = 0; i < DB_NUM; i++) db_imgs16[i] = (int_rgb**)IntColorAlloc2(db_height / 2, db_width / 2);
	db_height16 = db_height / 2;
	db_width16 = db_width / 2;
	MakeHalf(db_imgs32, DB_NUM, db_height, db_width, db_imgs16);										// db_imgs32 을 db_imgs16로 줄이기
	for (int i = 0; i < DB_NUM; i++) db_imgs8[i] = (int_rgb**)IntColorAlloc2(db_height / 4, db_width / 4);
	db_height8 = db_height / 4;
	db_width8 = db_width / 4;
	MakeHalf(db_imgs16, DB_NUM, db_height16, db_width16, db_imgs8);										// db_imgs16 을 db_imgs8로 줄이기
	set_img(db_imgs8, db_height8, db_width8, db_rgb_average, height, width, original_img, img_out);		//set_img() 함수 호출로 이미지  출력 
}


//4*4사이즈 db로 모자이크하기
void Mosaic4X4() {
	char db_dir[300] = "C:\\Users\\user\\Desktop\\김민정\\2020-2수업\\지능형영상처리\\솔루션\\지능형영상처리day1\\db_students";
	char db_fname_prefix[100] = "dbs";
	char db_fname[300];
	int db_height, db_width, db_height16, db_width16, db_height8, db_width8, db_height4, db_width4;		//추가로  db_height4, db_width4 선언
	int_rgb** db_imgs32[DB_NUM];
	int_rgb** db_imgs16[DB_NUM];
	int_rgb** db_imgs8[DB_NUM];
	int_rgb** db_imgs4[DB_NUM];																			//추가로 선언
	int_rgb db_rgb_average[DB_NUM];
	int height, width;
	char original_img_name[100] = "스파이더맨.jpg";
	int_rgb** original_img = ReadColorImage(original_img_name, &height, &width);
	if (height % 16 != 0) height -= height % 16;
	if (width % 16 != 0) width -= width % 16;
	int_rgb** img_out = (int_rgb**)IntColorAlloc2(height, width);
	for (int i = 0; i < DB_NUM; i++) {
		sprintf(db_fname, "%s\\%s%04d.jpg", db_dir, db_fname_prefix, i);
		db_imgs32[i] = ReadColorImage(db_fname, &db_height, &db_width);
	}
	for (int i = 0; i < DB_NUM; i++) db_imgs16[i] = (int_rgb**)IntColorAlloc2(db_height / 2, db_width / 2);
	db_height16 = db_height / 2;
	db_width16 = db_width / 2;
	MakeHalf(db_imgs32, DB_NUM, db_height, db_width, db_imgs16);										// db_imgs32 을 db_imgs16로 줄이기
	for (int i = 0; i < DB_NUM; i++) db_imgs8[i] = (int_rgb**)IntColorAlloc2(db_height / 4, db_width / 4);
	db_height8 = db_height / 4;
	db_width8 = db_width / 4;
	MakeHalf(db_imgs16, DB_NUM, db_height16, db_width16, db_imgs8);										// db_imgs16 을 db_imgs8로 줄이기
	for (int i = 0; i < DB_NUM; i++) db_imgs4[i] = (int_rgb**)IntColorAlloc2(db_height / 8, db_width / 8);
	db_height4 = db_height / 8;
	db_width4 = db_width / 8;
	MakeHalf(db_imgs8, DB_NUM, db_height8, db_width8, db_imgs4);										// db_imgs8 을 db_imgs4로 줄이기
	set_img(db_imgs4, db_height4, db_width4, db_rgb_average, height, width, original_img, img_out);		//set_img() 함수 호출로 이미지  출력 
}


//16*16사이즈 db로 랜덤 모자이크하기
void RANDOM_Mosaic16X16() {
	char db_dir[300] = "C:\\Users\\user\\Desktop\\김민정\\2020-2수업\\지능형영상처리\\솔루션\\지능형영상처리day1\\db_students";
	char db_fname_prefix[100] = "dbs";
	char db_fname[300];
	int db_height, db_width, db_height16, db_width16;																// db이미지의 높이, 너비 변수선언
	int_rgb** db_imgs32[DB_NUM];
	int_rgb** db_imgs16[DB_NUM];
	int_rgb db_rgb_average[DB_NUM];																					// db 이미지들의 평균 RGB를 담을 구조체형 배열
	int height, width;																								// 배경 영상의 높이와 너비 변수
	char original_img_name[100] = "스파이더맨.jpg";																	// 배경 영상의 파일 이름 ""안에 수정OK
	int_rgb** original_img = ReadColorImage(original_img_name, &height, &width);									// 배경영상(=원본)을 구조체로 읽어와 선언및 삽입
	if (height % 16 != 0) height -= height % 16;
	if (width % 16 != 0) width -= width % 16;
	int_rgb** img_out = (int_rgb**)IntColorAlloc2(height, width);													// 결과 영상 img_out	
	for (int i = 0; i < DB_NUM; i++) {
		sprintf(db_fname, "%s\\%s%04d.jpg", db_dir, db_fname_prefix, i);
		db_imgs32[i] = ReadColorImage(db_fname, &db_height, &db_width);
	}
	for (int i = 0; i < DB_NUM; i++)	db_imgs16[i] = (int_rgb**)IntColorAlloc2(db_height / 2, db_width / 2);		// 16*16 영상을 저장할 구조체의 메모리 할당
	MakeHalf(db_imgs32, DB_NUM, db_height, db_width, db_imgs16);													// db_imgs32 의 4픽셀을 하나의 픽셀로 합쳐 크기 줄인후  db_imgs16 배열에 저장
	db_height16 = db_height / 2;
	db_width16 = db_width / 2;
	set_img_RANDOM(db_imgs16, db_height16, db_width16, db_rgb_average, height, width, original_img, img_out);			//set_img_RANDOM() 함수 호출로 랜덤한 모자이크 이미지  출력 
}


void main()
{
	int height = 665;
	int width = 351;
	int_rgb** img = ReadColorImage("스파이더맨.jpg", &height, &width);
	int** img_out = (int**)IntAlloc2(height, width);
	ColorImageShow("입력영상", img, height, width); //원본영상 출력
	Mosaic16X16(); // db가 16*16인 모자이크 영상 출력
	Mosaic8X8();   //db가 8*8인 모자이크 영상 출력
	Mosaic4X4();   //db가 4*4인 모자이크 영상 출력
	RANDOM_Mosaic16X16(); // db가 16*16인 모자이크 영상 출력
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

