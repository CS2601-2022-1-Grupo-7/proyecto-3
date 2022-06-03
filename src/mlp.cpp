// Proyecto-3
// Copyright © 2022 Grupo 7
//
// proyecto-3 is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// proyecto-3 is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with hello.  If not, see <http://www.gnu.org/licenses/>.

#include "mlp.hpp"

VectorXd MLP::activation(VectorXd Net){//funcion de activacion
	for(size_t i = 0; i < Net.size(); i++)
	{
		Net(i) = 1.0/(1.0+exp(-Net(i)));
	}
	return Net;
}

MatrixXd MLP::derivada_ho(
	const MatrixXd& w,
	size_t k,
	VectorXd& delta,
	const VectorXd& So, // Output
	const VectorXd& Sd, // Desired
	const VectorXd& Shk // Hidden
	)
{

	delta.resize(w.cols());
	MatrixXd d = w;
	for(size_t i = 0; i < w.cols(); i++){
		delta(i) = (So(i)-Sd(i))*So(i)*(1.0-So(i));
	}

	for(size_t i = 0; i < w.rows(); i++){
		for(size_t j = 0; j < w.cols(); j++){
			d(i,j) = delta(j)*Shk(i);
		}
	}

	// std::cout << "d:" << std::endl;
	// std::cout << d << std::endl;
	return d;
}

MatrixXd MLP::derivada_hh(
	const MatrixXd& w,
	size_t km1,
	VectorXd &delta,
	const VectorXd& Shk, // Hidden
	const VectorXd& Shkm1 // Hidden
	)
{	
	
	delta.resize(w.cols());
	VectorXd tmp = delta;
	MatrixXd d = w;

	for(size_t j = 0; j < w.cols(); j++){
		for(size_t k = 0; k < w.rows(); k++){
			tmp(j) += delta(k)*w(j,k);
		}
		tmp(j) = tmp(j)*Shk(j)*(1.0-Shk(j)); 
	}

	for(size_t i = 0; i < w.rows(); i++){
		for(size_t j = 0; j < w.cols(); j++){
			d(i,j) = tmp(j)*Shkm1(i);			
		}
	}

	delta = tmp;
	return d;
}

VectorXd MLP::forward(VectorXd C)
{
	for(const auto& w: W)
	{
		C = activation(C.transpose()*w);
	}
	return C;
}

void MLP::backward(size_t epoch, double alpha, VectorXd So, VectorXd Sd, VectorXd Shk)
{
	//
	VectorXd delta;
	std::vector<MatrixXd> WT = W;
	while(epoch--)
	{
		for(ssize_t i = W.size()-1; i >= 0; i--)
		{
			if(i == W.size() -1)
				WT[i] -= alpha*derivada_ho(W[i], i, delta, So, Sd, Shk);
			else{
				WT[i] -= alpha*derivada_hh(W[i], i, delta, Shk,So);
			}
		}

		W = WT;
	}
}
