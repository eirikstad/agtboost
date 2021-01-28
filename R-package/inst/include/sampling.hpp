// sampling.hpp

#ifndef __SAMPLING_HPP_INCLUDED__
#define __SAMPLING_HPP_INCLUDED__

#include "external_rcpp.hpp"

Tmat<double> matrix_subset(Tmat<double> &x, Tvec<int> &ind){
    // Since only Eigen > 3.4 supports slicing by indices
    int n = ind.size();
    int m = x.cols();
    Tmat<double> msub(n, m);
    for(int i=0; i<n; i++)
    {
        msub.row(i) = x.row(ind[i]);
    }
    return msub;
}

Tvec<double> sample_vec(Tvec<double> &v, Tvec<int> &ind){
    return ind.unaryExpr(v);
}

Tvec<int> sample_int(int n, int size){
    // Uniformly shuffle vector of size 'n' and return head of size 'size'
    Tvec<int> ind(n);
    std::iota(ind.data(), ind.data()+ind.size(), 0);
    std::random_shuffle(ind.data(), ind.data()+ind.size());
    return ind.head(size); 
}

Tvec<int> sample_int_rate(int n, double sample_rate)
{
    int size = (int)(sample_rate * n); // the double is strictly positive
    return sample_int(n, size);
}


// void sample_illustration(Tvec<double> v, Tmat<double> x, double sample_rate){
//     // Downsample 'v' and 'm' with sample rate 'sample_rate'
//     int n = v.size(); 
//     int size = (int)(sample_rate * n); // the double is strictly positive
//     Tvec<int> ind = sample_int(n, size);
//     Tvec<double> vsub = ind.unaryExpr(v);
//     Rcpp::Rcout << "vector subset: \n" << vsub << std::endl;
//     Tmat<double> xsub = matrix_subset(x, ind);
//     Rcpp::Rcout << "matrix subset: \n" << xsub << std::endl;
// }

#endif