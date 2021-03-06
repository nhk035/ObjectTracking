//
// Created by nhk on 18-8-19.
//

#include "prepare.h"
#include <sys/time.h>

using namespace std;
using namespace cv;

prepare::prepare(const cv::Mat _image, const cv::Mat _mask) : mask(_mask) {

    CV_Assert(_image.rows == IMAGE_HEIGHT && _image.cols == IMAGE_WIDTH);

    //距离变换
//    threshold(mask, mask, 100, 255, THRESH_BINARY);
    //计算图像中每一个非零点距离离自己最近的零点的距离
    distanceTransform(~mask, distanceImage, CV_DIST_L2, CV_DIST_MASK_PRECISE);

    if (_image.channels() != 1)
        cvtColor(_image, image, COLOR_BGR2GRAY);

    cv::Mat hsvImage;
    cvtColor(_image, hsvImage, COLOR_BGR2HSV);
    image.create(hsvImage.size(), hsvImage.depth());
    int ch[] = {0, 0};
    mixChannels(&hsvImage, 1, &image, 1, ch, 1);


    //计算直方图
    cv::calcHist(&image, 1, &channels, mask, objHist, dims, &size, ranges);
    cv::calcHist(&image, 1, &channels, ~mask, bkgHist, dims, &size, ranges);
    normalize(objHist, objHist, 0, 255, NORM_MINMAX, -1, Mat());
    normalize(bkgHist, bkgHist, 0, 255, NORM_MINMAX, -1, Mat());

    ///直方图显示
//    int scale = 1;
//    Mat dstImage(size * scale, size, CV_8U, Scalar(0));
//    //获取最大最小值
//    double minValue = 0, maxValue = 0;
//    minMaxLoc(objHist, &minValue, &maxValue, 0, 0);
//    cout << minValue << " " << maxValue << endl;
//    int hpt = saturate_cast<int>(0.9 * size);
//    for (int i = 0; i < 256; i++) {
//        float binValue = objHist.at<float>(i);
//        int realValue = saturate_cast<int>(binValue * hpt / maxValue);
//        rectangle(dstImage, Point(i*scale, size-1), Point((i+1)*scale - 1, size - realValue), Scalar(255));
//    }
//    imshow("Hist", dstImage);


//    Mat image_obj, image_bkg;
//    image.copyTo(image_obj, mask);
//    image.copyTo(image_bkg, ~mask);


//    timeval time1, time2;
//    gettimeofday(&time1, NULL);
    generatePairs();
//    gettimeofday(&time2, NULL);
//    cerr << "compute neighbor pair cost: " << time2.tv_sec - time1.tv_sec + 0.000001*(time2.tv_usec - time1.tv_usec) << " s" << endl;

    g = new GraphType(/*estimated # of nodes*/ IMAGE_WIDTH*IMAGE_HEIGHT, /*estimated # of edges*/ pairs.size());

    g -> add_node(IMAGE_WIDTH*IMAGE_HEIGHT);
}



prepare::~prepare() {
    delete g;
}

int prepare::getNodeId(int row, int col){
    assert(col >= 0 && col < IMAGE_WIDTH && row >= 0 && row < IMAGE_HEIGHT);
    return row*IMAGE_WIDTH + col;
}

void prepare::generatePairs() {

    CV_Assert(image.rows == IMAGE_HEIGHT && image.cols == IMAGE_WIDTH);

    for (int i = 0; i < image.rows - 1; i++) {

        for (int j = 0; j < image.cols - 1; j++) {

            int downNode = getNodeId(i+1, j);
            int rightNode = getNodeId(i, j+1);
            int currentNode = getNodeId(i, j);
            pairs.emplace_back(pair<int, int>(currentNode, rightNode));
            pairs.emplace_back(pair<int, int>(currentNode, downNode));
        }
    }
    for (int j = 0; j < image.cols -1; ++j) {

        int i = image.rows -1;
        int rightNode = getNodeId(i, j+1);
        int currentNode = getNodeId(i, j);
        pairs.emplace_back(pair<int, int>(currentNode, rightNode));
    }
    for (int i = 0; i < image.rows - 1; ++i) {

        int j = image.cols - 1;
        int downNode = getNodeId(i+1, j);
        int currentNode = getNodeId(i, j);
        pairs.emplace_back(pair<int, int>(currentNode, downNode));
    }
}

void prepare::computeBoundaryTerm() {

    double aa = computeVariance();

    if (pairs.empty())
        cerr << "generate pairs first!" << endl;

    auto it = pairs.begin();
    for (; it != pairs.end(); ++it)
    {
        nLink boundary;
        boundary.first.row = getRowFromId(it->first);
        boundary.first.col = getColFromId(it->first);
        boundary.first.id = it->first;
        boundary.first.value = image.at<unsigned char>(boundary.first.row, boundary.first.col);

        boundary.second.row = getRowFromId(it->second);
        boundary.second.col = getColFromId(it->second);
        boundary.second.id = it->second;
        boundary.second.value = image.at<unsigned char>(boundary.second.row, boundary.second.col);

        boundary.weight = exp(-pow(boundary.first.value-boundary.second.value, 2)*0.5/aa);

        nLinks.push_back(boundary);
    }

}

void prepare::computeRegionBasedTerm() {


}

int prepare::getRowFromId(int id) {

    assert( id >= 0 && id < IMAGE_HEIGHT*IMAGE_WIDTH);
    return id/int(IMAGE_WIDTH);
}

int prepare::getColFromId(int id) {

    assert( id >= 0 && id < IMAGE_HEIGHT*IMAGE_WIDTH);
    return id%IMAGE_WIDTH;
}

unsigned char prepare::getValueFromId(int id) {

    assert( id >= 0 && id < IMAGE_HEIGHT*IMAGE_WIDTH);
    int row = getRowFromId(id);
    int col = getColFromId(id);
    return image.at<unsigned char>(row, col);
}

double prepare::computeVariance() {

    std::vector<std::thread> threads;

    //find out how many threads are supported, and how many pairs each thread will work on
    const unsigned numThreads = std::thread::hardware_concurrency() - 1;
    const unsigned numPairsForThread = (numThreads > pairs.size()) ? 1 : (unsigned)ceilf((float)(pairs.size()) / numThreads);

    std::mutex writeMutex;

//    cout << "Launch " << numThreads << " threads with " << numPairsForThread << " pairs per thread" << endl;

    vector<double> aa(numThreads, 0.);
    //invoke each thread with its pairs to process (if less pairs than threads, invoke only #pairs threads with 1 pair each)
    for (size_t threadId = 0; threadId < MIN(numThreads, pairs.size()); threadId++) {
        threads.push_back(std::thread([&, threadId] {
            const int startingPair = numPairsForThread * threadId;

            for (int j = 0; j < numPairsForThread; j++) {
                const int pairId = startingPair + j;
                if (pairId >= pairs.size()) { //make sure threads don't overflow the pairs
                    break;
                }
                const pair<int, int> pair = pairs[pairId];

                aa[threadId] +=  pow(getValueFromId(pair.first) - getValueFromId(pair.second), 2);


//                writeMutex.lock();
//                cout << "Thread " << threadId << ": Match (pair " << pairId << ") " << pair.first << ", " << pair.second << endl;
//                writeMutex.unlock();

            }
        }));
    }

    //wait for threads to complete
    for (auto& t : threads) {
        t.join();
    }

    double result = 0.;
    for (int i = 0; i < aa.size(); ++i) {
        result += aa[i];
    }
    return (result/pairs.size());

}

void prepare::update(const cv::Mat img) {


    assert(IMAGE_WIDTH == img.cols && IMAGE_HEIGHT == img.rows);
//    img.convertTo(image, cv::COLOR_BGR2GRAY);//注意这里的颜色顺序
    if (1 != img.channels())
        cvtColor(img, image, COLOR_BGR2GRAY);
    //使用hue色调通道数据
    cv::Mat hsvImage;
    cvtColor(img, hsvImage, COLOR_BGR2HSV);
    image.create(hsvImage.size(), hsvImage.depth());
    int ch[] = {0, 0};
    mixChannels(&hsvImage, 1, &image, 1, ch, 1);

    //计算反向投影图
    MatND backproj_obj, backproj_bkg;
    calcBackProject(&image, 1, &channels, objHist, backproj_obj, ranges,  1, true);
    calcBackProject(&image, 1, &channels, bkgHist, backproj_bkg, ranges,  1, true);


//    imshow("object反向投影图", backproj_obj);
//    imshow("background反向投影图", backproj_bkg);

//    waitKey(10);

//    timeval time1, time2;
//    gettimeofday(&time1, NULL);
    nLinks.clear();
    computeBoundaryTerm();
//    gettimeofday(&time2, NULL);
//    cerr << "compute boundary term cost: " << time2.tv_sec - time1.tv_sec + 0.000001*(time2.tv_usec - time1.tv_usec) << endl;

    g->reset();
    g->add_node(IMAGE_WIDTH*IMAGE_HEIGHT);

//    double minValue = 0, maxValue = 0;
//    minMaxLoc(distanceImage, &minValue, &maxValue, NULL, NULL);

    for (int i = 0; i < IMAGE_HEIGHT; ++i) {
        uchar* objData = backproj_obj.ptr<uchar>(i);
        uchar* bkgData = backproj_bkg.ptr<uchar>(i);
        float* disData = distanceImage.ptr<float>(i);
        for (int j = 0; j < IMAGE_WIDTH; ++j) {

            cerr << disData[j] << endl;
            double S = (objData[j] == 0 ? 6. : -log(objData[j]/255.0)) + disData[j];
            double T = (bkgData[j] == 0 ? 6. : -log(bkgData[j]/255.0));
            g -> add_tweights( i * IMAGE_WIDTH + j,   /* capacities */  S, T );
        }
    }

//    timeval time1, time2;
//    gettimeofday(&time1, NULL);
//    for (int i = 0; i < IMAGE_WIDTH*IMAGE_HEIGHT; ++i) {
//
//        double S = (backproj_obj.at<uchar>(i) == 0 ? 6. : -log(backproj_obj.at<uchar>(i)/255.0));
//        double T = (backproj_bkg.at<uchar>(i) == 0 ? 6. : -log(backproj_bkg.at<uchar>(i)/255.0));
//        g -> add_tweights( i,   /* capacities */  S, T );
//    }
//    gettimeofday(&time2, NULL);
//    cerr << "go through all pixel cost: " << time2.tv_sec - time1.tv_sec + 0.000001*(time2.tv_usec - time1.tv_usec) << endl;

//    timeval time3, time4;
//    gettimeofday(&time3, NULL);
    for (int j = 0; j < nLinks.size(); ++j) {
        g -> add_edge( nLinks[j].first.id, nLinks[j].second.id,    /* capacities */ 6*nLinks[j].weight, 6*nLinks[j].weight );
    }
//    gettimeofday(&time4, NULL);
//    cerr << "add edge cost: " << time4.tv_sec - time3.tv_sec + 0.000001*(time4.tv_usec - time3.tv_usec) << endl;


//    timeval time3, time4;
//    gettimeofday(&time3, NULL);
    float flow = g -> maxflow();
//    gettimeofday(&time4, NULL);
//    cerr << "compute max flow cost: " << time4.tv_sec - time3.tv_sec + 0.000001*(time4.tv_usec - time3.tv_usec) << endl;



    printf("Flow = %f\n", flow);
    printf("Minimum cut:\n");

    for (int k = 0; k < IMAGE_WIDTH*IMAGE_HEIGHT; ++k) {
        mask.at<uchar>(k) = (g->what_segment(k) == GraphType::SOURCE) ? 255 : 0;
    }
    distanceTransform(~mask, distanceImage, CV_DIST_L2, CV_DIST_MASK_PRECISE);
    imshow("mask", mask);

}
