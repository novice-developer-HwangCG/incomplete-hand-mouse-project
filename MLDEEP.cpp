#include <opencv2/opencv.hpp>		// opencv 헤더파일
#include <iostream>			// 메인헤더파일
using namespace cv;			// cv생략
using namespace std;			// std생략
using namespace cv::dnn;		// dnn 생략

const float CONFIDENCE_THRESHOLD = 0.5;	// 알고리즘 확신값
const float NMS_THRESHOLD = 0.5;			// 비 최대값 억제
const int NUM_CLASSES = 2;			// 클래스 2

const Scalar colors[] = {		// 색 행렬값
{0, 255, 255},
{255, 255, 0},
{0, 255, 0},
{255, 0, 0}
};
const auto NUM_COLORS = sizeof(colors) / sizeof(colors[0]);	// 색 변수형 크기 확인

int main() {		// 메인

	Mat frame, handimg;			// 행렬
	VideoCapture Cap(0);		// 영상
	if (!Cap.isOpened()) {		// 조건
		cout << "open failed" << endl;	// 실패문
		return -1;			// 비정상종료
	}

	vector<string> class_names = { "twofinger","threefinger" };		// 클래스 이름
	auto net = readNetFromDarknet("yolov4-finger.cfg", "yolov4-finger_final.weights");		// 학습,훈련데이터
	net.setPreferableBackend(DNN_BACKEND_OPENCV);		// 백엔드 지정
	net.setPreferableTarget(DNN_TARGET_CPU);					// 타깃 디바이스
	auto output_names = net.getUnconnectedOutLayersNames();	// 출력레이어이름

	namedWindow("frame_cap");		// 새 윈도우
	namedWindow("display");			// 새 윈도우

	while (1)		// 반복문
	{
		Cap >> frame;	// 영상 읽기
		if (frame.empty()) {		// 조건문
			break;		// 종료
		}
		Mat blob, conto;		// 행렬
		vector<Mat> detections;		// 벡터형 mat

		// MedianBlur, Morphology
		Mat open,close;
		morphologyEx(frame, open, MORPH_OPEN, Mat());
		morphologyEx(open, close, MORPH_CLOSE, Mat());
		Mat median_mask;
		medianBlur(close, median_mask, 3);

		
		// YCrCb
		Mat ycrcb;
		cvtColor(frame, ycrcb, COLOR_BGR2YCrCb);
		inRange(ycrcb, Scalar(0, 133, 77), Scalar(255, 173, 127), ycrcb);
		handimg = (ycrcb.size(), CV_8UC3, Scalar(0));
		add(frame, Scalar(0), handimg, ycrcb);
		
		//Contours
		Mat  contours_gray;
		cvtColor(handimg, contours_gray, COLOR_BGR2GRAY);
		vector<vector<Point>> contours;
		findContours(contours_gray, contours, RETR_EXTERNAL, CHAIN_APPROX_NONE);
		cvtColor(contours_gray, conto, COLOR_GRAY2BGR);
		for (int i = 0; i < contours.size(); i++) {
			Scalar c(0, 255, 0);
			drawContours(conto, contours, i, c, 2);
		}
		/* 컴파일 오류 contours의 외곽선의 최댓값
		int maxK = 0;
		double maxArea = contourArea(contours[0]);
		for (int k = 1; k < contours.size(); k++)
		{
			double area = contourArea(contours[k]);
			if (area > maxArea)
			{
				maxK = k;
				maxArea = area;
			}
		}
		vector<Point> handContour = contours[maxK];
		*/
		//Convexhull
		vector<vector<Point>>hull(contours.size());
		for (int i = 0; i < contours.size(); i++) {
			convexHull(Mat(contours[i]), hull[i], false);
			Scalar c(255, 0, 0);
			drawContours(conto, hull, i, c, 2);
		}
		/* 컴파일 오류 convexhull에서 최대값을 찾아 좌표를 저장하여 최대부분에 원을 그림
		vector<Vec4i> defects;
		convexityDefects(handContour, hull, defects);
		for (int k = 0; k < defects.size(); k++)
		{
			Vec4i v = defects[k];
			Point ptStart = handContour[v[0]];
			Point ptEnd = handContour[v[1]];
			Point ptFar = handContour[v[2]];
			float depth = v[3] / 256.0;
			if (depth > 10)
			{
				circle(conto, ptStart, 6, Scalar(0, 0, 255), 2);
				circle(conto, ptEnd, 6, Scalar(0, 0, 255), 2);
			}
		}
		*/
	
		// Yolo4 
		blobFromImage(frame, blob, 1 / 255.f, Size(320, 320), Scalar(), true, false, CV_32F);
		net.setInput(blob);
		net.forward(detections, output_names);
		vector<int> indices[NUM_CLASSES];
		vector<Rect> boxes[NUM_CLASSES];
		vector<float> scores[NUM_CLASSES];
		for (auto& output : detections)
		{
			const auto num_boxes = output.rows;
			for (int i = 0; i < num_boxes; i++)
			{
				auto x = output.at<float>(i, 0) * frame.cols;
				auto y = output.at<float>(i, 1) * frame.rows;
				auto width = output.at<float>(i, 2) * frame.cols;
				auto height = output.at<float>(i, 3) * frame.rows;
				Rect rect(x - width / 2, y - height / 2, width, height);
				for (int c = 0; c < NUM_CLASSES; c++)
				{
					auto confidence = *output.ptr<float>(i, 5 + c);
					if (confidence >= CONFIDENCE_THRESHOLD)
					{
						boxes[c].push_back(rect);
						scores[c].push_back(confidence);
					}
				}
			}
		}
		for (int c = 0; c < NUM_CLASSES; c++) {
			NMSBoxes(boxes[c], scores[c], 0.0, NMS_THRESHOLD, indices[c]);		// 경계상자
		}
		// 바운딩 박스, 클래스 표시
		for (int c = 0; c < NUM_CLASSES; c++)
		{
			for (int i = 0; i < indices[c].size(); ++i)
			{
				const auto color = colors[c % NUM_COLORS];
				auto idx = indices[c][i];
				const auto& rect = boxes[c][idx];
				rectangle(conto, Point(rect.x, rect.y), Point(rect.x + rect.width, rect.y + rect.height), color, 3);
				string label_str = class_names[c] + ": " + format("%.02lf", scores[c][idx]);
				int baseline;
				auto label_bg_sz = getTextSize(label_str, FONT_HERSHEY_COMPLEX_SMALL, 1, 1, &baseline);
				rectangle(conto, Point(rect.x, rect.y - label_bg_sz.height - baseline - 10), Point(rect.x + label_bg_sz.width, rect.y), color, FILLED);
				putText(conto, label_str, Point(rect.x, rect.y - baseline - 5), FONT_HERSHEY_COMPLEX_SMALL, 1, Scalar(0, 0, 0));
			}
		}

		int FPS = CAP_PROP_FPS;			// 프레임값
		String FPSstr = to_string(FPS);		// int to string
		putText(conto, FPSstr, Point(0, 25), FONT_HERSHEY_DUPLEX, 1, Scalar(255, 255, 255));		// 글자
		imshow("frame_cap", conto);		// 영상 출력

		if (waitKey(10) == 27) {		// ESC
			break;
		}
	}
	destroyAllWindows();		// 모든 윈도우 닫기
	return 0;			// 함수 종료
}

// 마우스 이벤트 삭제