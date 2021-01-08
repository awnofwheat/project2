//
//  CNN.hpp
//  project2
//
//  Created by 张嘉艺 on 2021/1/6.
//

#ifndef CNN_hpp
#define CNN_hpp

#include <stdio.h>

#include<opencv2/opencv.hpp>

class Matrix{
public:
    int channel;
    int row;
    int column;
    float* data;

    Matrix(int num_channels, int num_r, int num_c, float* data);
    ~Matrix();
    float getd (int channel, int r, int c) const;
    friend Matrix pool(const Matrix &in);
};
 
 
 



#endif /* CNN_hpp */
