//
// Created by nhk on 18-8-19.
//

#include "prepare.h"

using namespace std;
using namespace cv;

prepare::prepare(const cv::Mat _mask) :  mask(_mask) {


    image.zeros(IMAGE_HEIGHT, IMAGE_WIDTH, CV_8UC1);


    generatePairs();
}



prepare::~prepare() {

}

int prepare::getNodeId(int row, int col){
    assert(col >= 0 && col < IMAGE_WIDTH && row >= 0 && row < IMAGE_HEIGHT);
    return row*IMAGE_WIDTH + col;
}

void prepare::generatePairs() {

    for (int i = 0; i < image.rows - 1; ++i) {

        for (int j = 0; j < image.cols - 1; ++j) {

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

        double aa = computeVariance();
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

    cout << "Launch " << numThreads << " threads with " << numPairsForThread << " pairs per thread" << endl;

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


                writeMutex.lock();
                cout << "Thread " << threadId << ": Match (pair " << pairId << ") " << pair.first << ", " << pair.second << endl;
                writeMutex.unlock();

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
    cvtColor(img, image, COLOR_BGR2GRAY);


    int channels = 0;
    MatND dstHist;
    int dims = 1;
    int size = 256;
    float hranges[] = {0, 255};
    const float* ranges[] = {hranges};
    cv::calcHist(&image, 1, &channels, mask, dstHist, dims, &size, ranges);

    int scale = 1;
    Mat dstImage(size * scale, size, CV_8U, Scalar(0));
    //获取最大最小值
    double minValue = 0, maxValue = 0;
    minMaxLoc(dstHist, &minValue, &maxValue, 0, 0);
    int hpt = saturate_cast<int>(0.9 * size);
    for (int i = 0; i < 256; i++) {
        float binValue = dstHist.at<float>(i);
        int realValue = saturate_cast<int>(binValue * hpt / maxValue);
        rectangle(dstImage, Point(i*scale, size-1), Point((i+1)*scale - 1, size - realValue), Scalar(255));
    }
    imshow("Hist", dstImage);
    waitKey(0);

}
