//
// Created by nhk on 18-8-19.
///initialize
//选定object区域

///update
//确定obj区域与bkg区域的直方图
//在新的一帧中计算每个像素点标记到S或T的惩罚项
//计算邻域像素间不连续惩罚
//最大流最小割求解obj区域
//

#ifndef TRACKING_PREPARE_H
#define TRACKING_PREPARE_H

#include "common.h"
#include "maxflow/graph.h"
#include <math.h>
#include <map>
#include <vector>
#include <thread>
#include <mutex>

class prepare {

    typedef struct {

        int row;
        int col;
        int id;
        int value;
    }node;

    typedef struct {

        node first;
        node second;
        double weight;
    }nLink;

    typedef struct {

        int id;
        double weight;
    }tLink;

public:

    prepare(const cv::Mat mask);


    virtual ~prepare();

    void update(const cv::Mat img);

private:

    void generatePairs();

    int getNodeId(int row, int col);

    void computeBoundaryTerm();

    void computeRegionBasedTerm();

    int getRowFromId(int id);

    int getColFromId(int id);

    unsigned char getValueFromId(int id);

    /**
     * 使用多线程计算相邻像素值的方差
     */
    double computeVariance();



    cv::Mat image;
    cv::Mat mask;

    /**
     * save node id pair between all neighbor points
     */
    std::vector<std::pair<int, int> > pairs;

    std::vector<node> nodes;
    std::vector<nLink> nLinks;
    std::vector<tLink> tLinks;

    double lambda;

    unsigned obj[256] = {0};
    unsigned bkg[256] = {0};

};


#endif //TRACKING_PREPARE_H
