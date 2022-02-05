#include <iostream>
#include <math.h>
#include <vector>

using namespace std;

class in_x
{
public:
    float x1;
    float x2;

    in_x(float x_1,float x_2):
        x1(x_1),x2(x_2)
    {}
};


class Neural
{
public:
    Neural(float weight_1,float weight_2,float bia);
    float w1;
    float w2;
    float b;
    float z;
    float result;

    float sigmoid(float z);
    float sigmoid_Div(float z);
    float calculate(float in_1,float in_2);
};



Neural::Neural(float weight_1, float weight_2, float bia):
    w1(weight_1),
    w2(weight_2),
    b(bia),
    z(0),
    result(0)
{

}



float Neural::sigmoid(float z)
{
    return (1/(1+exp(-z)));
}



float Neural::sigmoid_Div(float z)
{
    float func = sigmoid(z);
    return func * (1 - func);
}


float Neural::calculate(float in_1,float in_2)
{
    z = in_1 * w1 + in_2 * w2 + b;
    result = sigmoid(z);
    return result;
}





class NetWork
{
public:
  NetWork(float w1,float w2,float b);
  float x1;
  float x2;
  Neural Hide_1;
  Neural Hide_2;
  Neural Out;

  float feedForword(float x_1, float x_2);
  float mse(std::vector<float> y_pre,std::vector<float>y_true);
  void feedBack(float y_true);

};

float NetWork::feedForword(float x_1,float x_2)
{
    Hide_1.calculate(x_1,x_2);
    Hide_2.calculate(x_1,x_2);
    Out.calculate(Hide_1.result,Hide_2.result);

    return Out.result;
}




float NetWork::mse(std::vector<float> y_pre_vec,std::vector<float>y_true_vec)
{

   float sum_err = 0.0f;
   for(size_t i = 0;i < y_pre_vec.size();i++)
   {
       sum_err += (y_pre_vec[i] - y_true_vec[i]) * (y_pre_vec[i] - y_true_vec[i]);
   }
   float mean_err =   sum_err/y_pre_vec.size();
   std::cout << "Loss :" << mean_err << std::endl;

   return mean_err;
}


void NetWork::feedBack(float y_true)
{
    float learnRate = 0.5f;

    //1.loss
    float Loss_yPre = -2.0f*(y_true - Out.result);

    //2.neural out
    // pre = sigmoid(w1 * Hide_1.result + w2 * Hide_2.result + bias)
    float Pre_Out_w1 = Out.sigmoid_Div(Out.z) * Hide_1.result;
    float Pre_Out_w2 = Out.sigmoid_Div(Out.z) * Hide_2.result;
    float Pre_Out_bia = Out.sigmoid_Div(Out.z);
    float Pre_Hid1 = Out.sigmoid_Div(Out.z) * Pre_Out_w1;
    float Pre_Hid2 =  Out.sigmoid_Div(Out.z) * Pre_Out_w2;


    //3.neural Hid1
    // Hide_1.result = sigmoid(w1 * x1 + w2 * x2 + bias)
    float Hide1_w1 = Hide_1.sigmoid_Div(Hide_1.z) * x1;
    float Hide1_w2 = Hide_1.sigmoid_Div(Hide_1.z) * x2;
    float Hide1_bias = Hide_1.sigmoid_Div(Hide_1.z);


    //4.neural Hid2
    // Hide_2.result = sigmoid(w1 * x1 + w2 * x2 + bias)
    float Hide2_w1 = Hide_2.sigmoid_Div(Hide_2.z) * x1;
    float Hide2_w2 = Hide_2.sigmoid_Div(Hide_2.z) * x2;
    float Hide2_bias = Hide_2.sigmoid_Div(Hide_2.z);




    Hide_1.w1 -= learnRate *  Loss_yPre * Pre_Hid1 * Hide1_w1;
    Hide_1.w2 -= learnRate *  Loss_yPre * Pre_Hid1 * Hide1_w2;
    Hide_1.b  -= learnRate *  Loss_yPre * Pre_Hid1 * Hide1_bias;

    Hide_2.w1 -= learnRate *  Loss_yPre * Pre_Hid2 * Hide2_w1;
    Hide_2.w2 -= learnRate *  Loss_yPre * Pre_Hid2 * Hide2_w2;
    Hide_2.b  -= learnRate *  Loss_yPre * Pre_Hid2 * Hide2_bias;

    Out.w1 -= learnRate * Loss_yPre * Pre_Out_w1;
    Out.w2 -= learnRate * Loss_yPre * Pre_Out_w2;
    Out.b -= learnRate * Loss_yPre * Pre_Out_bia;



}


NetWork::NetWork(float w1,float w2,float b):Hide_1(w1,w2,b),Hide_2(w1,w2,b),Out(w1,w2,b)
{

}



int main()
{


    std::vector<float> y_true = {1,0,0,1};
    std::vector<float> y_pre = {};

    std::vector<in_x> data;
    in_x x = in_x(-2,-1);
    data.push_back(x);
    x.x1 = 25;
    x.x2 = 6;
    data.push_back(x);
    x.x1 = 17;
    x.x2 = 4;
    data.push_back(x);
    x.x1 = -15;
    x.x2 = -6;
    data.push_back(x);

    NetWork Net(-10.5f,5.3f,6.0f);

    for(int i = 0;i < 1000;i++)
    {
        y_pre.clear();
        for(size_t i = 0; i < data.size();i++)
        {
            Net.feedForword(data[i].x1,data[i].x2);
            Net.feedBack(y_true[i]);
            y_pre.push_back(Net.Out.result);
            //std::cout << "y_pre:" << Net.Out.result << std::endl;
        }
        Net.mse(y_pre,y_true);
    }

    Net.feedForword(-7,-3);
    std::cout << "y_pre:(-7,-3)" << Net.Out.result << std::endl;
    Net.feedForword(20,2);
    std::cout << "y_pre:(20,2)" << Net.Out.result << std::endl;

    return 0;
}
