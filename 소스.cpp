#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <malloc.h>

#include <opencv2/opencv.hpp>   
#include <opencv2/core/core.hpp>   
#include <opencv2/highgui/highgui.hpp>  

using namespace cv;

#define PI 3.14159265359

typedef struct {
	int r, g, b;
}int_rgb;


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
	*no_label = connectedComponents(bw, labelImage, 8); // 0까지 포함된 갯수임

	(*no_label)--;

	for (int i = 0; i < height; i++) {
		for (int j = 0; j < width; j++)
			label[i][j] = labelImage.at<int>(i, j);
	}
}

void drawLine(int **image, int height, int width,double a, double b, double Thickness){

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
		
			double d = fabs(a*x - y + b) / sqrt(a*a + 1.0);

			if (d < Thickness) image[y][x] = 255;
		}
	}
}

void drawCircle(int **image, int height, int width, double a, double b, double Thickness) {

	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {
			double a1 = x - a;
			double b1 = y - b;
			double d = sqrt(a1*a1 + b1 *b1);

			if (d < Thickness) image[y][x] = 255;
		}
	}
}
//768 1024
void class_031() {
	int height, width;
	int** image = ReadImage("Koala.jpg", &height, &width);
	/*박스
	for (int y = 0; y < 500; y++) {
		for (int x = 0; x < 100; x++) {
			image[100+y][x + 200] = 255;*/


			/*image[y][x] : y=[0,767] x=[0,1023]
			for (int x = 0; x < width; x++) {
				int y = (int)(0.4*x + 100+0.5); //0.4는 플롯 int라서 짤림
				if (y > height - 1) continue;
				else {
					image[y][x] = 255;
				}
			}
			//선 두껍게하고싶으면 y를 아래위로 하나씩 더*/

			//근의공식
			//y = ax + b --> ax - y + b = 0 --> d = (ax0-y0+b)/sqrt(a*a+1)


			/*입력으로 줘야할것 : a,b,thickness, image, width, height
			double a = 3.0;
			double b = 50.0;
			double Thickness = 3.0;

			for (int y = 0; y < height; y++) {
				for (int x = 0; x < width; x++) {
					double d = fabs(a*x - y + b) / sqrt(a*a + 1.0);

					if (d < Thickness) image[y][x] = 255;
				}
			}
			원래함수*/


	double a = 50.0;
	double b = 50.0;
	double Thickness = 3;

	drawLine(image, height, width, a, b, Thickness);
	drawCircle(image, height, width, a, b, Thickness);


	//원의 방정식 (x-a)^2+(y-b)^2=r^2
	//r=sqrt((x-a)^2+(y-b)^2)



	ImageShow("test1", image, height, width);
}

void Affine_Transform(int** image, int** img_out, int height, int width, float a, float b, float c, float d, float t1, float t2) {
	for (int y = 0; y < height; y++) {
		for (int x = 0; x < width; x++) {

			int X = a*x + b*y + t1 + 0.5;
			int Y = c*x + d*y +t2 +0.5;
			if(X<1024 && Y<768)
				img_out[Y][X] = image[y][x];
		}
	}
	ImageShow("test1", img_out, height, width);
}

//Affine Transform 
void class_0313() {

	int height, width;
	int** image = ReadImage("Koala.jpg", &height, &width);

	float a, b, c, d;
	a = 1; b = 0; c = 0; d = 2;

	int ** Affine_image = IntAlloc2(height, width);

	Affine_Transform(image, Affine_image, height, width, a, b, c, d,0,0);

}

#define imax(x, y) ((x)>(y) ? x : y)
#define imin(x, y) ((x)<(y) ? x : y)

int BilinearInterpolation(int** image, int width, int height, double x, double y) // 선형보간
{	
	int x_int = (int)x;
	int y_int = (int)y;

	int A = image[imin(imax(y_int, 0), height - 1)][imin(imax(x_int, 0), width - 1)];
	int B = image[imin(imax(y_int, 0), height - 1)][imin(imax(x_int + 1, 0), width - 1)];
	int C = image[imin(imax(y_int + 1, 0), height - 1)][imin(imax(x_int, 0), width - 1)];
	int D = image[imin(imax(y_int + 1, 0), height - 1)][imin(imax(x_int + 1, 0), width - 1)];

	double dx = x - x_int;
	double dy = y - y_int;

	double value
		= (1.0 - dx)*(1.0 - dy)*A + dx*(1.0 - dy)*B
		+ (1.0 - dx)*dy*C + dx*dy*D;

	return((int)(value + 0.5));
}

void upgrade_Affine(int** image, int** img_out, int height, int width, float a, float b, float c, float d, float t1, float t2) {
	float s = 1/(a*d - b*c);
	
	for (int y_prime = 0; y_prime < height; y_prime++) {
		for (int x_prime = 0; x_prime < width; x_prime++) {
			double x = s*(d*x_prime  - d*t1 - b*y_prime + b*t2) + width/2;
			double y = s*(-c*x_prime + c*t1 + a*y_prime - a*t2) + height/2;
			if(x< (double)width && x>=0.0 && y< (double)height && y>=0.0)
				img_out[y_prime][x_prime] = BilinearInterpolation(image, width, height, x, y);
		}
	}

	ImageShow("test1", image, height, width);
	ImageShow("test2", img_out, height, width);

}

void class0320() {
	int height, width;

	int** image = ReadImage("Koala.jpg", &height, &width);
	
	int** image_out = IntAlloc2(height, width);
	
	float a, b, c, d;
	a = 1 / sqrt(2) ; b = 1 ; c = -1 ; d = 1 / sqrt(2) ;
	//a = -1; b = 0; c = 0; d = 1;

	upgrade_Affine(image, image_out, height, width, a, b, c, d, width/2, height/2);

	//ImageShow("1", image, height, width);
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
typedef struct POS3D{
	float x;
	float y;
	float z;
};

void init_POS3D(POS3D* P, float x, float y, float z) {
	P->x = x;
	P->y = y;
	P->z = z;
}

float eval_len_POS3D(POS3D P1, POS3D P2) {
	float len = 0;

	len = pow(P1.x - P2.x, 2) + pow(P1.y - P2.y, 2) + pow(P1.y - P2.y, 2);
	len = sqrt(len);

	return len;
}


void drawLine_pinhole(int** image, int** image_out, int height, int width, POS3D Px0, POS3D Px1, float f1, float f2 ) {
	float X, Y, Z;
	float x, y;
	
	float px = width / 2; float py = height / 2; float w = 0;
	float len = eval_len_POS3D(Px0, Px1);
	//l = p+t(q-p) 200,300

	for (float t = 0.1; t < len; t =t+ len/10000) {
		X = Px0.x + t*(Px1.x - Px0.x);
		Y = Px0.y + t*(Px1.y - Px0.y);
		Z = Px0.z + t*(Px1.z - Px0.z);

		x = f1*X + px*Z;
		y = f2*Y + py*Z;
		w = Z;
		if ((x/w)< (double)width && (x / w) >= 0.0 && (y / w)< (double)height && (y / w) >= 0.0)
		 image_out[int((y / w))][int((x / w))] = 255;
	}

	ImageShow("aa", image_out, height, width);



}

/*
 핀홀 카메라: (4,2)가 어느 점에 매칭되는지 알고 싶으면 뒤에 숫자 2 로 나누면 (2,1) <- 
 3차원의 경우: (4,6,2)->(2,3,1) ; 3차원의 공간의 그림자: 2차원 ; projective space
*/
int main() {
	int height, width;

	int** image = ReadImage("Koala.jpg", &height, &width);
	int** image_out1 = IntAlloc2(height, width);
	int** image_out2 = IntAlloc2(height, width);

	//ImageShow("qq", image, height, width);

	POS3D Px0, Px1;

	init_POS3D(&Px0, 1, 2, 3);
	init_POS3D(&Px1, 2, 3, 1);

	drawLine_pinhole(image, image_out1, height, width, Px0,Px1, 200, 300);
	
	
}