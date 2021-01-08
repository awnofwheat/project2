//
//  CNN.cpp
//  project2
//
//  Created by 张嘉艺 on 2021/1/6.
//

#include "Matrix.hpp"


#include <fstream>
#include <cassert>

using namespace cv;
using namespace std;
Matrix::Matrix(int num_channels, int num_r, int num_c, float* data){
    this->channel = num_channels;
    this->row = num_r;
    this->column = num_c;
    this->data = data;
}
Matrix::~Matrix(){
     delete[] data;
}

float Matrix::getd (int channel, int r, int c)const{
    if(r < 0 || r > row-1 || c < 0 || c > column-1){
        return 0;
    }
    else
    {
        return data[channel*column*row + r*column + c];
    }
}





