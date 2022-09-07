#include <vector>
#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <assert.h>
using std::cout;
using std::endl;
using std ::vector;

int RandomNumber()
{
    return (rand() % 100);
}


void verify_result(const vector<float> &a, const vector<float> &b,const vector<float> &c,int numElements)
{
    // For every row...
    for (int i = 0; i < numElements; i++)
    {
        // For every column...
        for (int j = 0; j < numElements; j++)
        {
            // For every element in the row-column pair
            float tmp = 0;
            for (int k = 0; k < numElements; k++)
            {
                // Accumulate the partial results
                tmp += a[i * numElements + k] * b[k * numElements + j];
            }
            // Check against the CPU result
            
            assert((tmp - c[i * numElements + j]) < 0.01);
            //std::cout<<tmp<<"   "<<c[i * numElements + j]<<std::endl;
        }
    }
}