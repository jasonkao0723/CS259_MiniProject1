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


void compare(VTYPE* neuron1, VTYPE* neuron2, int Oy, int Ox, int Oz) {
    bool correct = true;

    for (int y = 0; y < Oy; y++) {
        for (int x = 0; x < Ox; x++) {
            for (int z = 0; z < Oz; z++) {
                if (fabs(neuron1[y * (Ox * Oz) + x * (Oz)+z] - neuron2[y * (Ox * Oz) + x * (Oz)+z]) > TOL) {
                    printf("Error at (y:%d, x:%d, z:%d) \t Host result: %lf \t Device result: %lf\n", y, x, z, neuron1[y * (Ox * Oz) + x * (Oz)+z], neuron2[y * (Ox * Oz) + x * (Oz)+z]);
                    correct = false;
                }
                //printf("Error at (y:%d, x:%d, z:%d) \t Host result: %lf \t Device result: %lf\n", y, x, z, neuron1[y * (Ox * Oz) + x * (Oz)+z], neuron2[y * (Ox * Oz) + x * (Oz)+z]);


            }
        }
    }
    if (correct) {
        std::cout << "RESULTS MATCH!" << std::endl;
    }


}

void* aligned_malloc(uint64_t align, uint64_t bytes) {
    size_t mask = (align - 1) ^ ((size_t)-1);
    char* ptr = (((char*)malloc(bytes + align)) + align);
    ptr = (char*)(((size_t)ptr) & mask);
    return (void*)ptr;
}

#endif