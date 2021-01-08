//
//  main.cpp
//  cnn
//
//  Created by 张嘉艺 on 2021/1/6.
//


#include<iostream>
#include "face_binary_cls.hpp"
#include<cmath>
#include <chrono>
#include"Matrix.hpp"
#include <iomanip>
using namespace std;
using namespace cv;



Matrix conv(conv_param &co, const Matrix &in){
    int out_size_r = (in.row+co.pad*2-co.kernel_size)/co.stride+1;
    int out_size_c = (in.column+co.pad*2-co.kernel_size)/co.stride+1;
    float *out_data = new float[co.out_channels*out_size_r*out_size_c];
    for (int o = 0; o < co.out_channels; o++){
        float bias = co.p_bias[o];
        float kernels[co.in_channels*co.kernel_size*co.kernel_size];
        for (int i = 0; i < co.in_channels; i++){
            for (int k = 0; k < co.kernel_size*co.kernel_size; k++){
                int idx = o*co.kernel_size*co.kernel_size*co.in_channels + i*co.kernel_size*co.kernel_size + k;
                kernels[i*co.kernel_size*co.kernel_size+k] = co.p_weight[idx];
            }
        }
        for (int r = 0; r < out_size_r; r++){
            for (int c = 0; c < out_size_c; c++){
                float sum = bias;
                int init_r = r*co.stride-co.pad;
                int init_c = c*co.stride-co.pad;
                for (int i = 0; i < co.in_channels; i++){
                    for (int k1 = 0; k1 < co.kernel_size; k1++){
                        for (int k2 = 0; k2 < co.kernel_size; k2++){
                            sum += in.getd(i,init_r+k1,init_c+k2)*kernels[i*co.kernel_size*co.kernel_size+k1*co.kernel_size+k2];
                        }
                    }
                }
                if(sum < 0)
                    sum = 0;
                out_data[o*out_size_c*out_size_r+r*out_size_c+c] = sum;
            }
        }
    }
    Matrix out = Matrix(co.out_channels, out_size_r, out_size_c, out_data);
    return out;
}

Matrix optimize(conv_param &co, const Matrix &in){
    int in_size_r = in.row;
    int in_size_c = in.row;
    int out_size_r = (in_size_r+co.pad*2-co.kernel_size)/co.stride+1;
    int out_size_c = (in_size_c+co.pad*2-co.kernel_size)/co.stride+1;
    float *out_data = new float[co.out_channels*out_size_r*out_size_c];
    for (int o = 0; o < co.out_channels; o++){
        float bias = co.p_bias[o];
        float kernels[co.in_channels*co.kernel_size*co.kernel_size];
        for (int i = 0; i < co.in_channels; i++){
            for (int k = 0; k < co.kernel_size*co.kernel_size; k++){
                int idx = o*co.kernel_size*co.kernel_size*co.in_channels + i*co.kernel_size*co.kernel_size + k;
                kernels[i*co.kernel_size*co.kernel_size+k] = co.p_weight[idx];
            }
        }
        for (int r = 0; r < out_size_r; r++){
            for (int c = 0; c < out_size_c; c++){
                float sum = bias;
                int init_r = r*co.stride-co.pad;
                int init_c = c*co.stride-co.pad;
                for (int i = 0; i < co.in_channels; i++){
                    for (int k1 = 0; k1 < co.kernel_size; k1++){
                        for (int k2 = 0; k2 < co.kernel_size; k2++){
                            sum += in.getd(i,init_r+k1,init_c+k2)*kernels[i*co.kernel_size*co.kernel_size+k1*co.kernel_size+k2];
                        }
                    }
                }
                if(sum < 0)
                    sum = 0;
                out_data[o*out_size_c*out_size_r+r*out_size_c+c] = sum;
            }
        }
    }
    Matrix out = Matrix(co.out_channels, out_size_r, out_size_c, out_data);
    return out;
}

Matrix pool(const Matrix &in){
    int num_channels = in.channel;
    int out_size_r = in.row/2;
    int out_size_c = in.column/2;
    float * out_data = new float[out_size_c*out_size_r*num_channels];
    for (size_t i = 0; i < num_channels; i++){
        for (size_t r = 0; r < out_size_r; r++){
            for (size_t c = 0; c < out_size_c; c++){
                float max = in.getd(i,2*r,2*c);
                if(in.getd(i,2*r,2*c+1) > max){
                    max = in.getd(i,2*r,2*c+1);
                }
                if(in.getd(i,2*r+1,2*c) > max){
                    max = in.getd(i,2*r+1,2*c);
                }
                if(in.getd(i,2*r+1,2*c+1) > max){
                    max = in.getd(i,2*r+1,2*c+1);
                }
                out_data[r*out_size_c+c+i*out_size_c*out_size_r] = max;
            }
        }
    }
    Matrix out = Matrix(num_channels, out_size_r, out_size_c, out_data);
    return out;
}

void test(string filename){
    
    auto start = std::chrono::steady_clock::now();
    auto end = std::chrono::steady_clock::now();
    auto duration = 0L;
    start=std::chrono::steady_clock::now();
    Mat img = imread(filename);
//    Mat3f image_f(image);
    float *in = new float[3*128*128];
    //img/=255;
    for (int i = 0; i < 128; i++)
    {
        for (int j = 0; j < 128; j++)
        {
            in[0*128*128+128*i+j] = (float)img.at<Vec3b>(i,j).val[0]/255;
            in[1*128*128+128*i+j] = (float)img.at<Vec3b>(i,j).val[1]/255;
            in[2*128*128+128*i+j] = (float)img.at<Vec3b>(i,j).val[2]/255;
        }
    }
    

    Matrix in0(3,128,128,in);
    const Matrix &out0 = conv(conv_params[0],in0);
    const Matrix &out0_pool = pool(out0);
    const Matrix &out1 = conv(conv_params[1],out0_pool);
    const Matrix &out1_pool = pool(out1);
    const Matrix &out2 = conv(conv_params[2],out1_pool);
    int out_features = fc_params[0].out_features;
    int in_features = fc_params[0].in_features;
    float fc_out[out_features];
    float* conv2_out = out2.data;

    for (int o = 0; o < 2; o++) {
        float sum = 0;
        for (int i = 0; i < 2048; i++) {
            float w_oi = fc_params[0].p_weight[o*2048+i];
            sum += w_oi*conv2_out[i];
        }
        float bias = fc_params[0].p_bias[o];
        sum += bias;
        fc_out[o] = sum;
    }
    float* proba=new float[2];
    float escore=0;
    for(int i=0;i<2;i++){
        escore+=exp(fc_out[i]);
    }
    for(int i=0;i<2;i++){
        *(proba+i)=exp(fc_out[i])/escore;
    }
    end=std::chrono::steady_clock::now();
    duration=std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
    cout <<  setiosflags(ios::fixed) << setprecision(6) <<"bg score:"<<proba[0]<<", "<<"face score:" << proba[1]<< endl;
    cout<<duration<<"ms"<<endl;
}

void test_opt(string filename){
    auto start = std::chrono::steady_clock::now();
    auto end = std::chrono::steady_clock::now();
    auto duration = 0L;
    start=std::chrono::steady_clock::now();
    Mat img = imread(filename);
//    Mat3f image_f(image);
    float *in = new float[3*128*128];
    //img/=255;
    for (int i = 0; i < 128; i++)
    {
        for (int j = 0; j < 128; j++)
        {
            in[0*128*128+128*i+j] = (float)img.at<Vec3b>(i,j).val[0]/255;
            in[1*128*128+128*i+j] = (float)img.at<Vec3b>(i,j).val[1]/255;
            in[2*128*128+128*i+j] = (float)img.at<Vec3b>(i,j).val[2]/255;
        }
    }
    

    Matrix in0(3,128,128,in);
    const Matrix &out0 = optimize(conv_params[0],in0);
    const Matrix &out0_pool = pool(out0);
    const Matrix &out1 = optimize(conv_params[1],out0_pool);
    const Matrix &out1_pool = pool(out1);
    const Matrix &out2 = optimize(conv_params[2],out1_pool);
    int out_features = fc_params[0].out_features;
    int in_features = fc_params[0].in_features;
    float fc_out[out_features];
    float* conv2_out = out2.data;

    for (int o = 0; o < 2; o++) {
        float sum = 0;
        for (int i = 0; i < 2048; i++) {
            float w_oi = fc_params[0].p_weight[o*2048+i];
            sum += w_oi*conv2_out[i];
        }
        float bias = fc_params[0].p_bias[o];
        sum += bias;
        fc_out[o] = sum;
    }
    
    float* proba=new float[2];
    float escore=0;
    for(int i=0;i<2;i++){
        escore+=exp(fc_out[i]);
    }
    for(int i=0;i<2;i++){
        *(proba+i)=exp(fc_out[i])/escore;
    }
    end=std::chrono::steady_clock::now();
    duration=std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
    cout <<  setiosflags(ios::fixed) << setprecision(6) <<"bg score:"<<proba[0]<<", "<<"face score:" << proba[1]<< endl;
    cout<<duration<<"ms"<<endl;
    
}

int main(){
    test("./samples/face.jpg");
    test_opt("./samples/face.jpg");
    test("./samples/bg.jpg");
    test_opt("./samples/bg.jpg");

    return 0;
}

