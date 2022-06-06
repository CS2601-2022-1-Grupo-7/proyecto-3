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

MatrixXd MLP::derivada_ho(
	const MatrixXd& w,
	size_t k,
	VectorXd& delta,
	const VectorXd& So, // Output
	const VectorXd& Sd, // Desired
	const VectorXd& Shk // Hidden
	)
{

	delta.resize(So.size());
	MatrixXd d(w.rows(), w.cols());
	for(size_t i = 0; i < So.size(); i++){
		delta(i) = ((So(i)-Sd(i))*So(i)*(1.0-So(i)));
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

	VectorXd tmp(w.cols());
	MatrixXd d(w.rows(), w.cols());

	for(size_t j = 0; j < w.cols(); j++){
		double valTmp = 0.0;
		for(size_t k = 0; k < delta.size(); k++){
			valTmp += delta(k)*W[km1+1](j,k);
		}
		tmp(j) = valTmp*Shk(j)*(1.0-Shk(j));
	}

	for(size_t i = 0; i < w.rows(); i++){
		for(size_t j = 0; j < w.cols(); j++){
			d(i,j) = tmp(j)*Shkm1(i);
		}
	}

	delta = tmp;
	return d;
}

VectorXd MLP::softMax(VectorXd So){
	VectorXd result(So.size());
	double sum=0;
	for(int i=0; i<So.size(); i++){
		sum+=exp(So(i));
	}
	for(int i=0; i<So.size(); i++){
		result(i)=exp(So(i))/sum;
	}
	return result;
}

VectorXd MLP::forward(VectorXd C, std::vector<VectorXd>& Sh)
{
	Sh.push_back(C);
	for(const auto& w: W)
	{
		C = activation(C.transpose()*w);
		Sh.push_back(C);
	}
	return softMax(C);
}

VectorXd class2vector(int _class, size_t n)
{
	VectorXd v(n);

	for(size_t i = 0; i < n; i++)
	{
		v[i] = (int)i == _class-1 ? 1 : 0;
	}

	return v;
}

void MLP::backward(size_t epoch, double alpha, VectorXd x, int y)
{
	std::vector<VectorXd> Sh;
	VectorXd So = forward(x, Sh);
	VectorXd Sd = class2vector(y, W.back().cols());
	

	VectorXd delta(W[W.size()-1].cols()); //cambiar
	std::vector<MatrixXd> WT = W;
	while(epoch--)
	{
		for(ssize_t i = W.size()-1; i >= 0; i--)
		{

			//output,desired,hk -> derivada_ho
			//hk,hkm1 -> derivada_hh 

			if(i == W.size() -1){
				WT[i] -= alpha*derivada_ho(W[i], i, delta, Sh[i+1], Sd, Sh[i]);
			}
			else{
				WT[i] -= alpha*derivada_hh(W[i], i, delta, Sh[i+1], Sh[i]);
			}
		}

		W = WT;
	}
}

MLP::MLP(size_t features,
		size_t hidden_layers,
		size_t output_neurons,
		const std::vector<size_t>& neurons,
		std::function<VectorXd(const VectorXd&)> activation):
		activation(activation)
{
	if (hidden_layers != 0){
		MatrixXd m = MatrixXd::Random(features,neurons[0]);
		W.push_back(m);
		for (int i=0; i<hidden_layers-1; i++){
			int row = neurons[i];
			int col = neurons[i+1];
			m = MatrixXd::Random(row, col);
			W.push_back(m);
		}
		m = MatrixXd::Random(neurons[neurons.size()-1],output_neurons);
		W.push_back(m);
	}else{
		MatrixXd m = MatrixXd::Random(features, output_neurons);
		W.push_back(m);
	}
};
