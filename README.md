# 002
该项目完成了完整的特征点提取以及筛选和匹配的步骤

1. 特征点提取：由于针对的是动态目标，需要大量特征点以供筛选，设置特征点最大数量为3000，使用ORB特征法 \
2. 动态特征点剔除：根据语义掩码去除动态目标上的特征点 \
3. 距离筛选：计算最大最小距离，设置最大筛选距离为最小汉明距离的两倍 \
4. RANSAC精筛选：对匹配点对进行精筛选， 剔除外点，被剔除的外点为绿色，绘制在原图中 \
