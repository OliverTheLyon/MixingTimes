//
// Created by Oliver Lyon on 2024-04-09.
//

#include <Eigen/Dense>
#include <iostream>

#ifndef MIXINGTIMES_MATRIXWARPPER_H
#define MIXINGTIMES_MATRIXWARPPER_H

using Eigen::MatrixXd;
using Eigen::VectorXd;

namespace MarkovMix {
    class MM_Matrix{
    private:
        //vars
        MatrixXd        mat;                    //governs the switching process in CTMC or DTMC
        VectorXd        pi;
        bool            is_DTMC;
        bool            is_reversable;
        bool            is_ergodic;             //MIGHT BE PROBLEMATIC!!! MIGHT REMOVE LATER
        const double    err_thresh = 1e-15;

        //set up methods
        bool safety_check(MatrixXd* input_matrix){
            if(input_matrix->rows() != input_matrix->cols()){
                throw std::invalid_argument("Invalid matrix provided: (Must be a square matrix; rows = " +\
                std::to_string(input_matrix->rows()) + " ,cols = " +std::to_string(input_matrix->cols()) + ")");
            }

            //check the sum of each row is (1 DTMC, p_mat) or (0 CTMC, r_mat)
            VectorXd vec = VectorXd::Constant(input_matrix->rows(), 1, 1.0);
            vec = (*input_matrix) * vec;

            if(!(abs(vec.maxCoeff() - vec.minCoeff()) < err_thresh)){
                throw std::invalid_argument("Invalid matrix provided: (The rows do not sum consistently: Max = "+\
                std::to_string(vec.maxCoeff()) + ", Min = " +std::to_string(vec.minCoeff())+ ")");
            }

            //see if discrete  or continuous time
            if(abs(vec.maxCoeff() - 1) < err_thresh){
                for(int idx_row = 0; idx_row < input_matrix->rows(); idx_row++) {
                    for(int idx_col = 0; idx_col < input_matrix->cols(); idx_col++) {
                        if(0.0 > (*input_matrix)(idx_row, idx_col)) {
                            throw std::invalid_argument("Invalid discrete time matrix provided: (A negative probability found, "+\
                            std::to_string((*input_matrix)(idx_row, idx_col)) + ")");
                        }
                    }
                }
                is_DTMC = true;
                calc_stationary_dist_discrete(&mat, &pi);
            }
            else if(abs(vec.maxCoeff()) < err_thresh){
                for(int idx_row = 0; idx_row < input_matrix->rows(); idx_row++) {
                    for(int idx_col = 0; idx_col < input_matrix->cols(); idx_col++) {
                        if(idx_row != idx_col){
                            if(0.0 > (*input_matrix)(idx_row, idx_col)) {
                                throw std::invalid_argument("Invalid discrete time matrix provided: (A negative probability found, "+\
                            std::to_string((*input_matrix)(idx_row, idx_col)) + ")");
                            }
                        }
                        else{
                            if(0.0 < (*input_matrix)(idx_row, idx_col)) {
                                throw std::invalid_argument("Invalid discrete time matrix provided: (A negative probability found, "+\
                            std::to_string((*input_matrix)(idx_row, idx_col)) + ")");
                            }
                        }

                    }
                }
                is_DTMC = false;
                calc_stationary_dist_continuous(&mat, &pi);
            }
            else{
                throw std::invalid_argument("Invalid matrix provided: (The rows sum to values that are inconsistent\
 with a discrete time (1.0) or continuous time (0.0) Markov model: Each row sums to " +\
                std::to_string(vec.maxCoeff()));
            }
        }

        void calc_stationary_dist_continuous(MatrixXd* A, VectorXd* pi){
            //pi A = 0
            //sum(pi) = 1
            unsigned size = A->rows();
            MatrixXd A_prime(size,size + 1);
            A_prime << (*A) , MatrixXd::Ones(size,1);
            VectorXd b = VectorXd::Constant(size+1, 1, 0.0); //target solution
            b(size) = 1.0;
            *pi = ((A_prime).transpose()).fullPivHouseholderQr().solve(b);

            for(int i = 0 ; i < pi->size() ; i++){
                if((*pi)(i) < 0){
                    for(int j = 0 ; j < pi->size() ; j++){
                        if((*pi)(j) < 0){
                            (*pi)(j) = 0;
                        }
                    }
                    (*pi) /= pi->sum();
                }
            }
        }

        void calc_stationary_dist_discrete(MatrixXd* A, VectorXd* pi){
            // pi P = pi
            // P - I = 0
            // sum(pi) = 1
            unsigned size = A->rows();
            MatrixXd A_prime(size,size + 1);
            A_prime << (*A) - MatrixXd::Identity(size,size) , MatrixXd::Ones(size,1);
            VectorXd b = VectorXd::Constant(size+1, 1, 0.0); //target solution
            b(size) = 1.0;
            *pi = ((A_prime).transpose()).fullPivHouseholderQr().solve(b);

            for(int i = 0 ; i < pi->size() ; i++){
                if((*pi)(i) < 0){
                    for(int j = 0 ; j < pi->size() ; j++){
                        if((*pi)(j) < 0){
                            (*pi)(j) = 0;
                        }
                    }
                    (*pi) /= pi->sum();
                }
            }
        }

//
//        bool check_nonergod_stat(MutSel* mod){
//            double branch_length = 1.0;
//            VectorXd pi(61);
//            MatrixXd mutsel_spec;
//            mutsel_spec = (*mod->get_rates_matrix());
//            calc_stationary_dist(&mutsel_spec,&pi);
//
//            for(int j = 0 ; j < mutsel_spec.rows() ; j++){
//                if(pi(j) <= 1e-50){
//                    return(true);
//                }
//            }
//            return(false);
//        }
//
//        bool check_nonergod(MutSel* mod){
//            double branch_length = 10.0;
//            MatrixXd mutsel_spec, mutsel_ev_inv;
//            mutsel_spec = ((*(mod->get_eigen_vals())) * branch_length).array().exp().matrix().asDiagonal();
//            mutsel_ev_inv = (*(mod->get_eigen_vecs())).fullPivHouseholderQr().solve(MatrixXd::Identity(mutsel_spec.rows(),mutsel_spec.rows()));
//            mutsel_spec = (*(mod->get_eigen_vecs())) * mutsel_spec * mutsel_ev_inv;
//
//            for(int j = 0 ; j < mutsel_spec.rows() ; j++){
//                for(int l = 0 ; l < mutsel_spec.rows() ; l++){
//                    if(mutsel_spec(j,l) <= 0){
//                        return(true);
//                    }
//                }
//            }
//            return(false);
//        }
//
//        double KLD(VectorXd pi1, VectorXd pi2){
//            double kld = 0;
//            for(int i = 0 ; i < 61 ; i++){
//                if(pi1(i) <= 0 && pi2(i) <= 0){
//                    kld += 0;
//                }
//                else if(pi1(i) <= 0 && pi2(i) > 0){
//                    kld += 1e-50 * log2(1e-50/pi2(i));
//                }
//                else if(pi1(i) > 0 && pi2(i) <= 0){
//                    kld += pi1(i) * log2(pi1(i)/1e-50);
//                }
//                else {
//                    kld += pi1(i) * log2(pi1(i) / pi2(i));
//                }
//            }
//            return(kld);
//        }
//
//        double JSD(VectorXd pi1, VectorXd pi2){
//            return((KLD(pi1,pi2) + KLD(pi2, pi1))/2);
//        }





    public:
        //constructors
        MM_Matrix(MatrixXd rates_matrix){
            mat = rates_matrix;                  //rates matrix
            pi = VectorXd(rates_matrix.cols()); //stationary dist
            safety_check(&rates_matrix);         //check properties
        }


        bool is_discrete_time(){
            return(is_DTMC);
        }

        VectorXd get_stationary_distribution(){
            return(pi);
        }

    };

}
#endif //MIXINGTIMES_MATRIXWARPPER_H
