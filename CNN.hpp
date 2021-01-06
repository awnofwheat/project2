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
    float operator() (int channel, int r, int c) const;
    float* flatten() const;
    friend Matrix maxPool2x2(const Matrix &in);
};
 
 
 



#endif /* CNN_hpp */
