#include "common.h"
#include "maxflow/graph.h"
#include "prepare.h"

#define WINDOW_NAME "tracking"

cv::Rect g_rectangle;
bool g_bDrawingBox = false;
cv::RNG g_rng(12345);

const cv::Scalar RED = cv::Scalar(0,0,255);
const cv::Scalar PINK = cv::Scalar(230,130,255);
const cv::Scalar BLUE = cv::Scalar(255,0,0);
const cv::Scalar LIGHTBLUE = cv::Scalar(255,255,160);
const cv::Scalar GREEN = cv::Scalar(0,255,0);

void DrawRectangle(cv::Mat& img, cv::Rect box) {

    //随机颜色
//    cv::rectangle(img, box.tl(), box.br(), cv::Scalar(g_rng.uniform(0, 255), g_rng.uniform(0, 255), g_rng.uniform(0, 255)));
    cv::rectangle( img, box.tl(), box.br(), GREEN, 2);
}

void on_MouseHandle(int event, int x, int y, int flags, void* param) {

    cv::Mat& image = *(cv::Mat*)param;
    switch(event)
    {
        //鼠标移动消息
        case cv::EVENT_MOUSEMOVE:
        {
            if (g_bDrawingBox) {
                g_rectangle.width = x - g_rectangle.x;
                g_rectangle.height = y - g_rectangle.y;
            }
        }
            break;

            //左键按下
        case cv::EVENT_LBUTTONDOWN:
        {
            g_bDrawingBox = true;
            g_rectangle = cv::Rect(x, y, 0, 0);//记录起始点
        }
            break;

        case cv::EVENT_LBUTTONUP:
        {
            g_bDrawingBox = false;

            if (g_rectangle.width < 0)
            {
                g_rectangle.x += g_rectangle.width;
                g_rectangle.width *= -1;
            }
            if (g_rectangle.height < 0)
            {
                g_rectangle.y += g_rectangle.height;
                g_rectangle.height *= -1;
            }

            DrawRectangle(image, g_rectangle);
        }
            break;
    }
}



void ShowHelpText() {

}


int main()
{
    typedef Graph<float,float,float> GraphType;
    GraphType *g = new GraphType(/*estimated # of nodes*/ 2, /*estimated # of edges*/ 1);

    g -> add_node();
    g -> add_node();

    g -> add_tweights( 0,   /* capacities */  5, 0 );
    g -> add_tweights( 1,   /* capacities */  0, 5 );
    g -> add_edge( 0, 1,    /* capacities */  0.1, 0.1 );

    float flow = g -> maxflow();

    printf("Flow = %f\n", flow);
    printf("Minimum cut:\n");
    if (g->what_segment(0) == GraphType::SOURCE)
        printf("node0 is in the SOURCE set\n");
    else
        printf("node0 is in the SINK set\n");
    if (g->what_segment(1) == GraphType::SOURCE)
        printf("node1 is in the SOURCE set\n");
    else
        printf("node1 is in the SINK set\n");

    delete g;

    g_rectangle = cv::Rect(-1, -1, 0, 0);
    cv::Mat img(IMAGE_WIDTH, IMAGE_HEIGHT, CV_8UC3);

    cv::namedWindow(WINDOW_NAME, cv::WINDOW_NORMAL);
    cv::setMouseCallback(WINDOW_NAME, on_MouseHandle, (void*)&img);

    cv::VideoCapture cap(0);
    if (!cap.isOpened())
        exit(-1);
    cap.set(CV_CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH);
    cap.set(CV_CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT);
    cap.set(CV_CAP_PROP_FPS, 30);

    prepare* pp;
    while(cap.grab()){

        cap >> img;
        cv::Mat mask;

        if (g_bDrawingBox){

            cv::Mat tempImg;
            img.copyTo(tempImg);
            DrawRectangle(tempImg, g_rectangle);
            cv::imshow("initialSeg", tempImg);



        }
        if (g_rectangle.area() > 0 && !g_bDrawingBox){

            mask = cv::Mat::zeros(img.size(), CV_8UC1);
            cv::Mat I(g_rectangle.size(), CV_8UC1, cv::Scalar::all(255));
            I.copyTo( mask(g_rectangle) );
//            std::cout << mask << std::endl;
            pp = new prepare(mask);
            cv::imshow("mask", mask);
        }


        pp->update(img);


        cv::imshow(WINDOW_NAME, img);
//        std::cout << img.channels() << std::endl;
        cv::waitKey(10);
    }


    return 0;
}

