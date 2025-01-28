//
// Created by Oliver Lyon on 2025-01-24.
//
//g++ -I /usr/local/include/eigen3/ -o test main.cpp

#include "MatrixWarpper.h"
#include <iostream>
#include <Eigen/Dense>


int main(){

    //TEST1
    MatrixXd discrete_test_matrix_1 = MatrixXd::Ones(3,3);
    try {
        MarkovMix::MM_Matrix test(discrete_test_matrix_1);
        std::cout<<"Test 1 Failed: Invalid Matrix Caught."<<std::endl;
    }
    catch(...){
        std::cout<<"Test 1 passed: Invalid Matrix Caught."<<std::endl;
    }

    //TEST2
    discrete_test_matrix_1 = discrete_test_matrix_1 * -1.0;
    try {
        MarkovMix::MM_Matrix test(discrete_test_matrix_1);
        std::cout<<"Test 2 Failed: Invalid Matrix Caught."<<std::endl;
    }
    catch(...){
        std::cout<<"Test 2 passed: Invalid Matrix Caught."<<std::endl;
    }

    //TEST3,4
    discrete_test_matrix_1 = discrete_test_matrix_1 / -3.0;
    MarkovMix::MM_Matrix test(discrete_test_matrix_1);
    if(test.is_discrete_time()){
        std::cout<<"Test 3 passed: Matrix IDed as discrete."<<std::endl;
    }
    else{
        std::cout<<"Test 3 failed:  Matrix IDed as continuous."<<std::endl;
        exit(1);
    }
    if(abs(test.get_stationary_distribution().sum() - 1.0) < 1e-10){
        std::cout<<"Test 4 passed: Stationary distribution."<<std::endl;
    }
    else{
        std::cout<<"Test 4 failed: Stationary distribution."<<std::endl;
        exit(1);
    }

    //TEST5,6
    MatrixXd discrete_test_matrix_2(3,3);
    discrete_test_matrix_2 <<   0.5,    0.25,   0.25,
                                0.1,    0.8,    0.1,
                                0.05,   0.05,   0.9;
    MarkovMix::MM_Matrix test2(discrete_test_matrix_2);
    if(abs(test.get_stationary_distribution().sum() - 1.0) < 1e-10){
        std::cout<<"Test 5 passed: Stationary distribution."<<std::endl;
    }
    else{
        std::cout<<"Test 5 failed: Stationary distribution."<<std::endl;
        exit(1);
    }
    if(test2.get_stationary_distribution()(0) < test2.get_stationary_distribution()(1)) {
        if (test2.get_stationary_distribution()(1) < test2.get_stationary_distribution()(2)) {
            std::cout << "Test 6 passed: Stationary distribution inequality." << std::endl;
        }
    }
    else{
        std::cout<<"Test 6 failed: Stationary distribution inequality."<<std::endl;
        exit(1);
    }




}
