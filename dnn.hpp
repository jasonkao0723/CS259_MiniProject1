#pragma once
#ifndef DNN_H
#define DNN_H

#include <inttypes.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
//#include <sys/time.h>
#define TOL 1e-3
#define VTYPE float

// Is this a leaky relu?
VTYPE transfer(VTYPE i) {
    return i;
}


void compare(VTYPE* neuron1, VTYPE* neuron2, int Oy, int Ox, int Oz, int B) {
    bool correct = true;
    
    for (int b = 0; b < B; b++) 
    {
         for (int z = 0; z < Oz; z++)
        {
            for (int y = 0; y < Oy; y++)
            {
                for (int x = 0; x < Ox; x++)
                {
                    VTYPE n1 = neuron1[b * (Ox * Oy * Oz) + z * (Ox * Oy) + y * Ox + x];
                    VTYPE n2 = neuron2[b * (Ox * Oy * Oz) + z * (Ox * Oy) + y * Ox + x];
                    if (fabs(n1 - n2) > TOL) {
                        printf("Error at (y:%d, x:%d, z:%d, b:%d) \t Host result: %lf \t Device result: %lf\n", y, x, z, b, n1, n2);
                        correct = false;
                    }
                    //printf("Error at (y:%d, x:%d, z:%d, b:%d) \t Host result: %lf \t Device result: %lf\n", y, x, z, b, n1, n2);

                }
            }
        }
    }
    
    if (correct) {
        std::cout << "RESULTS MATCH!" << std::endl;
    }
    else {
        std::cout << "RESULTS DONT MATCH!!" << std::endl;
    }

}

void* aligned_malloc(uint64_t align, uint64_t bytes) {
    size_t mask = (align - 1) ^ ((size_t)-1);
    char* ptr = (((char*)malloc(bytes + align)) + align);
    ptr = (char*)(((size_t)ptr) & mask);
    return (void*)ptr;
}

#endif
