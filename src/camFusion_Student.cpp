
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <pcl/segmentation/extract_clusters.h>
#include <set>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        // pixel coordinates
        pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0); 
        pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0); 

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}

/* 
* The show3DObjects() function below can handle different output image sizes, but the text output has been manually tuned to fit the 2000x2000 size. 
* However, you can make this function work for other sizes too.
* For instance, to use a 1000x1000 size, adjusting the text positions by dividing them by 2.
*/
void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}

double calcEuclideanDistance(cv::Point2f& point1, cv::Point2f& point2){
    return sqrt(pow((point1.x - point2.x),2) + pow((point1.y - point2.y), 2));
}

// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    vector<double> euclidanDistance; 
    map<cv::DMatch, double> matchedDistance;
    for(auto kptmatch : kptMatches){
        auto prevKpt = kptsPrev[kptmatch.queryIdx];
        auto currKpt = kptsCurr[kptmatch.trainIdx];
        if(boundingBox.roi.contains(currKpt.pt)){
            auto dist = calcEuclideanDistance(prevKpt.pt, currKpt.pt);
            euclidanDistance.push_back(dist);
            matchedDistance[kptmatch] = dist; 
        }

    }

    sort(euclidanDistance.begin(), euclidanDistance.end());
    double medianDistance ;
    if(euclidanDistance.size() % 2 == 0){
        medianDistance = (euclidanDistance[euclidanDistance.size() / 2] + euclidanDistance[(euclidanDistance.size() / 2) - 1]) / 2.0;
    }else{
        medianDistance = euclidanDistance[euclidanDistance.size() / 2];
    }

    for(auto& match : matchedDistance){
        if(match.second >= medianDistance * 0.5 && match.second <= medianDistance * 1.50){
            boundingBox.kptMatches.push_back(match.first);
        }
    }
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    constexpr double minDistance = 100.0;
    vector<double> disRatios;

    for(auto itr = kptMatches.begin(); itr != kptMatches.end(); ++itr){
        auto kptCurr = kptsCurr.at(itr->trainIdx);
        auto kptPrv = kptsPrev.at(itr->queryIdx);

        for(auto itr2 = kptMatches.begin() + 1 ; itr2 != kptMatches.end(); ++itr2){
            auto kptCurr2 = kptsCurr.at(itr2->trainIdx);
            auto kptPrv2 = kptsPrev.at(itr2->queryIdx);

            double distCurr = cv::norm(kptCurr.pt - kptCurr2.pt);
            double distPrv = cv::norm(kptPrv.pt - kptPrv2.pt);

            if(distPrv >= minDistance && distPrv > std::numeric_limits<double>::epsilon()){
                disRatios.push_back(distCurr / distPrv);
            }
        }
    }

    if(disRatios.size() == 0){
        TTC = NAN;
        return;
    }
    sort(disRatios.begin(), disRatios.end());
    double medianDistanceRatio;
    if(disRatios.size() % 2 == 0){
        medianDistanceRatio = (disRatios[disRatios.size() / 2] + disRatios[(disRatios.size() / 2) - 1]) / 2.0;
    }else{
        medianDistanceRatio = disRatios[disRatios.size() / 2];
    }

    TTC = (-1.0 / frameRate) / (1 - medianDistanceRatio);
}

// remove outliers
pcl::PointCloud<pcl::PointXYZ>::Ptr euclidanClustering(std::vector<LidarPoint> &lidarPoints){
    
    pcl::PointCloud<pcl::PointXYZ>::Ptr outputCloud(new pcl::PointCloud<pcl::PointXYZ>());
    pcl::PointCloud<pcl::PointXYZ>::Ptr inputCloud(new pcl::PointCloud<pcl::PointXYZ>());
    for(auto lidarPoint : lidarPoints){
        inputCloud->push_back(pcl::PointXYZ((float)lidarPoint.x, (float)lidarPoint.y, (float)lidarPoint.z));
    }
    //  Create Kd-Tree for neighbour points search
    pcl::search::KdTree<pcl::PointXYZ>::Ptr searchTree(new pcl::search::KdTree<pcl::PointXYZ>());
    searchTree->setInputCloud(inputCloud);

    vector<pcl::PointIndices> indices;
    pcl::EuclideanClusterExtraction<pcl::PointXYZ> ecuClustering;
    ecuClustering.setInputCloud(inputCloud);
    ecuClustering.setSearchMethod(searchTree);
    ecuClustering.setMinClusterSize(4);
    ecuClustering.setClusterTolerance(0.05);
    ecuClustering.extract(indices);

    for(auto indice : indices){
        for(auto index : indice.indices){
            outputCloud->points.push_back(inputCloud->points[index]);
        }
    }

    return outputCloud;

}
void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    auto prvCloud = euclidanClustering(lidarPointsPrev);
    auto currCloud = euclidanClustering(lidarPointsCurr);

    if(prvCloud->points.size() != 0 && currCloud->points.size() != 0){
        float minPrv = prvCloud->points[0].x;
        for(auto point : prvCloud->points){
            minPrv = min(minPrv, point.x);
        }   

        float minCurr = currCloud->points[0].x;
        for(auto point : currCloud->points){
            minCurr = min(minCurr, point.x);
        }

        TTC = minCurr * (1 / frameRate) / (minPrv - minCurr);
    }else{
        TTC = NAN;
    } 
}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    map<vector<int>, int> b2bmatchCounts;
    for(auto& match : matches){
        auto prvFramePoint = prevFrame.keypoints[match.queryIdx].pt;
        auto currFramePoint = currFrame.keypoints[match.queryIdx].pt;

        for(auto prvFrameBox : prevFrame.boundingBoxes){
            if(prvFrameBox.roi.contains(prvFramePoint)){
                for(auto currFrameBox : currFrame.boundingBoxes){
                    ++b2bmatchCounts[{prvFrameBox.boxID, currFrameBox.boxID}];
                }
            }
        }
    }

    set<int> matchedPrvFrameBoxIds;
    set<int> matchedCurrFrameBoxIds;
    for(auto b2bmatch : b2bmatchCounts){
        if(matchedPrvFrameBoxIds.find(b2bmatch.first[0]) == matchedPrvFrameBoxIds.end()){
            matchedPrvFrameBoxIds.insert(b2bmatch.first[0]);
            matchedCurrFrameBoxIds.insert(b2bmatch.first[1]);
            bbBestMatches.insert({b2bmatch.first[0], b2bmatch.first[1]});
        }
    }
}
