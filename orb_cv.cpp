#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <chrono>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

int main(int argc, char **argv) {
  //-- 读取图像
  Mat image1 = imread("./1341846435.414309.png", CV_LOAD_IMAGE_COLOR);//连续帧匹配,相邻帧  walking_halfsphere
  Mat image2 = imread("./1341846435.450323.png", CV_LOAD_IMAGE_COLOR);
  Mat label1 = imread("./seg/1341846435.414309.png",CV_LOAD_IMAGE_COLOR);
  Mat label2 = imread("./seg/1341846435.450323.png", CV_LOAD_IMAGE_COLOR);

  Mat img_1 = image1.clone();
  Mat img_2 = image2.clone();
  Mat lab_1 = label1.clone();
  Mat lab_2 = label2.clone();

  assert(img_1.data != nullptr && img_2.data != nullptr);

  //-- 初始化
  std::vector<KeyPoint> keypoints1, keypoints2;
  std::vector<KeyPoint> keypoints_1, keypoints_2; //语义筛选后的特征点
  Mat descriptors_1, descriptors_2;
  Ptr<FeatureDetector> detector = ORB::create(3000);
  Ptr<DescriptorExtractor> descriptor = ORB::create();
  Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");

  //-- 第一步:检测 Oriented FAST 角点位置
  chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
  detector->detect(img_1, keypoints1);
  detector->detect(img_2, keypoints2);
  for(int a = 0; a < keypoints1.size(); a++)
  {
       if(lab_1.at<Vec3b>(keypoints1[a].pt.y,keypoints1[a].pt.x)[2]!=190)
        keypoints_1.push_back(keypoints1[a]);
  }
  for(int b = 0; b < keypoints2.size(); b++)
  {
      if(lab_2.at<Vec3b>(keypoints2[b].pt.y,keypoints2[b].pt.x)[2]!=190)
      keypoints_2.push_back(keypoints2[b]);
  }

  //-- 第二步:根据角点位置计算 BRIEF 描述子
  descriptor->compute(img_1, keypoints_1, descriptors_1);
  descriptor->compute(img_2, keypoints_2, descriptors_2);
  chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
  chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "extract ORB cost = " << time_used.count() << " seconds. " << endl;

  Mat lab1;
  drawKeypoints(img_1, keypoints_1, lab1, Scalar(0,0,255), DrawMatchesFlags::DEFAULT);
  imshow("no person points-1",lab1);
  Mat lab2;
  drawKeypoints(img_2, keypoints_2, lab2, Scalar(0,0,255), DrawMatchesFlags::DEFAULT);
  imshow("no person points-2",lab2);
  waitKey(0==27);

  //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
  vector<DMatch> matches;
  t1 = chrono::steady_clock::now();
  matcher->match(descriptors_1, descriptors_2, matches);
  t2 = chrono::steady_clock::now();
  time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);
  cout << "match ORB cost = " << time_used.count() << " seconds. " << endl;

  //-- 第四步:匹配点对筛选
  // 计算最小距离和最大距离
  auto min_max = minmax_element(matches.begin(), matches.end(),
                                [](const DMatch &m1, const DMatch &m2) { return m1.distance < m2.distance; });
  double min_dist = min_max.first->distance;
  double max_dist = min_max.second->distance;

  printf("-- Max dist : %f \n", max_dist);
  printf("-- Min dist : %f \n", min_dist);

  //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
  std::vector<DMatch> good_matches;
  for (int i = 0; i < descriptors_1.rows; i++) {
    if (matches[i].distance <= max(2 * min_dist, 30.0)) {
      good_matches.push_back(matches[i]);
    }
  }
  //-- 第五步:绘制匹配结果
  Mat img_match;
  Mat img_goodmatch;
  drawMatches(img_1, keypoints_1, img_2, keypoints_2, matches, img_match,Scalar(255,0,0), Scalar(0,0,255));
  drawMatches(img_1, keypoints_1, img_2, keypoints_2, good_matches, img_goodmatch,Scalar(255,0,0), Scalar(0,0,255));
  imshow("origin matches", img_match);
  waitKey(0==27);
  imshow("matches after hamming distance", img_goodmatch);
  waitKey(0==27);
  //精匹配——RANSAC///////////////////////////////////////////////////////////////////////////////////////////////////////
  std::vector<Point2f> obj1;
  std::vector<Point2f> scene1;
  for (size_t i = 0; i < good_matches.size(); ++i)
  {
        // 获取距离筛选后的匹配结果
        obj1.push_back(  keypoints_1[ good_matches[i].queryIdx ].pt); //queryidx是匹配的点对中，imgA的序列号，.pt就是它的位置
        scene1.push_back(  keypoints_2[ good_matches[i].trainIdx ].pt);  //trainidx疑似imgB中与imgA对应的匹配点序列号，.pt就是它的位置

  }  //obj1对应scene1就是匹配点对的位置
  cv::Mat H1= findHomography( obj1, scene1, CV_RANSAC,4 );
    //findHomography——计算二维点对间的最优变换矩阵，使用策略为RANSAC，4为将点对视为内点的最大允许重投影错误阈值
    //应该这个阈值越小，重投影矩阵计算的越精确，一般设置在1-10之间
    std::vector<Point2f> obj_corners1(4);  //用矩阵储存了img_1四角的四个点
    obj_corners1[0] = cvPoint(0,0);
    obj_corners1[1] = cvPoint( img_1.cols, 0);
    obj_corners1[2] = cvPoint( img_1.cols, img_1.rows);
    obj_corners1[3] = cvPoint( 0, img_1.rows);
    std::vector<Point2f> scene_corners1(4);

    perspectiveTransform( obj_corners1, scene_corners1, H1);  //利用H1进行投影变换，拉住imgA的四角，变换到imgB上



    int ptCount1 = (int)good_matches.size();//记录粗匹配点对的个数
    // 把<vector>matches转换为Mat
    //2列的MAT矩阵，每一行都存储了匹配点对的位置，x和y
    Mat p11(ptCount1, 2, CV_32F);
    Mat p21(ptCount1, 2, CV_32F);
    Point2f pt1; //工具人
    for (int i=0; i<ptCount1; i++)
    {
        pt1 = keypoints_1[good_matches[i].queryIdx].pt;
        p11.at<float>(i, 0) = pt1.x;
        p11.at<float>(i, 1) = pt1.y;

        pt1 = keypoints_2[good_matches[i].trainIdx].pt;
        p21.at<float>(i, 0) = pt1.x;
        p21.at<float>(i, 1) = pt1.y;
    }//将匹配点对的位置转存到位置矩阵中

    // 用RANSAC方法计算F
    Mat m_Fundamental1;
    // 上面这个变量是基本矩阵
    vector<uchar> m_RANSACStatus1;
    // 上面这个变量已经定义过，用于存储RANSAC后每个点的状态
    findFundamentalMat(p11, p21, m_RANSACStatus1, FM_RANSAC);//FM_LMEDS FM_RANSAC
    //RANSAC是通过特征点对的位置信息，来计算基础矩阵，从而去除外点

    // 计算野点个数
    int OutlinerCount1 = 0;
    for (int i=0; i<ptCount1; i++)
    {
        if (m_RANSACStatus1[i] == 0) // 状态为0表示野点
        {
            OutlinerCount1++;
        }
    }

    // 计算内点
    vector<Point2f> m_LeftInlier1;
    vector<Point2f> m_RightInlier1;
    vector<Point2f> m_LeftOutlier1;
    vector<Point2f> m_RightOutlier1;
    vector<DMatch> m_InlierMatches1;
    // 上面三个变量用于保存内点和匹配关系
    int InlinerCount1 = ptCount1 - OutlinerCount1; //先去除外点，内点个数=匹配点-外点
    m_InlierMatches1.resize(InlinerCount1);
    m_LeftInlier1.resize(InlinerCount1);
    m_RightInlier1.resize(InlinerCount1);
    m_LeftOutlier1.resize(OutlinerCount1);
    m_RightOutlier1.resize(OutlinerCount1);
    InlinerCount1 = 0;
    OutlinerCount1 = 0;
    for (int i=0; i<ptCount1; i++)
    {
        if (m_RANSACStatus1[i] != 0)
        {
            m_LeftInlier1[InlinerCount1].x = p11.at<float>(i, 0);
            m_LeftInlier1[InlinerCount1].y = p11.at<float>(i, 1);
            m_RightInlier1[InlinerCount1].x = p21.at<float>(i, 0);
            m_RightInlier1[InlinerCount1].y = p21.at<float>(i, 1);
            m_InlierMatches1[InlinerCount1].queryIdx = InlinerCount1;
            m_InlierMatches1[InlinerCount1].trainIdx = InlinerCount1;
            InlinerCount1++;
        }
        else
        {
            m_LeftOutlier1[OutlinerCount1].x = p11.at<float>(i, 0);
            m_LeftOutlier1[OutlinerCount1].y = p11.at<float>(i, 1);
            m_RightOutlier1[OutlinerCount1].x = p21.at<float>(i, 0);
            m_RightOutlier1[OutlinerCount1].y = p21.at<float>(i, 1);
            OutlinerCount1++;
        }
    }//将内点转存到矩阵中，LeftInlier储存原图中的内点位置，RightInlier储存匹配图中的内点位置，InlierMatches储存的是内点点对的匹配关系
    //内点点对的匹配关系包括：内点点对index，第一对内点，第二对内点。。。。。。

    // 把内点转换为drawMatches可以使用的格式
    vector<KeyPoint> key11(InlinerCount1);
    vector<KeyPoint> key21(InlinerCount1);
    vector<KeyPoint> key31(OutlinerCount1);
    vector<KeyPoint> key41(OutlinerCount1);
    KeyPoint::convert(m_LeftInlier1, key11);
    KeyPoint::convert(m_RightInlier1, key21);
    KeyPoint::convert(m_LeftOutlier1, key31);
    KeyPoint::convert(m_RightOutlier1, key41);//按照内点索引和内点点对位置，重新构建特征点
    Mat OutImage1;
    drawMatches(img_1, key11, img_2, key21, m_InlierMatches1, OutImage1,Scalar(255,0,0), Scalar(255,0,0),vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    //drawMatches参数里需要的matches，看来不需要存储距离信息，只需要对应点对的索引和自身序号匹配即可
    Mat chayi1 = img_1.clone();
    Mat chayi2 = img_2.clone();
    drawKeypoints(img_1,key31, chayi1, Scalar(0, 255, 0));
    drawKeypoints(img_2,key41, chayi2, Scalar(0, 255, 0));
    imshow( "RANSAC", OutImage1 );
    printf("去除外点:%d\n",OutlinerCount1);
    printf("所有保留特征点数:%d\n",m_InlierMatches1.size());
    waitKey(0==27);
    imshow("differences of RANSAC-1",chayi1);
    waitKey(0==27);
    imshow("differences of RANSAC-2",chayi2);
    waitKey(0==27);
  return 0;
}
