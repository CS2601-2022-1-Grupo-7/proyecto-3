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
#include "utils.hpp"

MatrixXd MLP::derivada_ho(
	const MatrixXd& w,
	size_t k,
	VectorXd& delta,
	const VectorXd& So, // Output --- Softmax Probabilidads
	const VectorXd& Sd, // Desired
	const VectorXd& Shk // Hidden --- Softmax Output
	)
{

	MatrixXd d(w.rows(), w.cols());
	delta.resize(So.size());

	for(size_t i = 0; i < So.size(); i++){
		delta(i) = ((So[i]-Sd[i])*So[i]*(1.0-So[i]));
	}

	for(size_t i = 0; i < w.rows(); i++){
		for(size_t j = 0; j < w.cols(); j++){
			d(i,j) = delta(j)*Shk(i);
		}
	}

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

void MLP::forward(VectorXd C, double b)
{
	this->Sh.push_back(C);
	for(const auto& w: W)
	{
		// std::cout << "multi:\n" << C.transpose()*w << std::endl;
		C = activation(C.transpose()*w);
		// for(int c=0; c<C.size(); c++){
		// 	C[c] = C[c]+b;
		// }
		this->Sh.push_back(C);
	}
	// std::cout<<"C\n" << C << std::endl;
	// this->Sh.push_back(softMax(C));
	// this->Sh.push_back(C);
}

double calc_E(const VectorXd& Sd, const VectorXd& So)
{
	double E = 0.0;

	for(size_t i = 0; i < So.size(); i++)
	{
		E += (Sd[i]*log(So[i]));
	}
	return -1.0*E;
}

std::tuple<VectorXd, double> MLP::testing(VectorXd C, int y, double b){
	// for (auto i=0; i< W.size(); i++){
	// 	std::cout << W[i] << std::endl;
	// }
	VectorXd Sdd = class2vector(y, W.back().cols());

	for(const auto& w: W)
	{
		C = activation(C.transpose()*w);
		for(int c=0; c<C.size(); c++){
			C[c] = C[c]+b;
		}
	}
	return {C, calc_E(Sdd, C)};
}


double MLP::training(double alpha, VectorXd x, int y, double bias){
	// for (auto i=0; i< W.size(); i++){
	// 	std::cout << W[i] << std::endl;
	// }

	VectorXd Sd = class2vector(y, W.back().cols());

	forward(x, bias);
	backward(alpha, y);
	forward(x, bias);

	return calc_E(Sd, Sh.back());
}

void MLP::backward(double alpha, int y)
{
	VectorXd Sd = class2vector(y, W.back().cols());
	VectorXd delta(W[W.size()-1].cols()); //cambiar
	std::vector<MatrixXd> WT = W;

	for(ssize_t i = W.size()-1; i >= 0; i--)
	{
		if(i == W.size() -1){
			WT[i] -= alpha*derivada_ho(W[i], i, delta, Sh[i+1], Sd, Sh[i+0]);
		}
		else{
			WT[i] -= alpha*derivada_hh(W[i], i, delta, Sh[i+0], Sh[i]);
		}
	}

	W = WT;
}

MLP::MLP(size_t features,
		size_t hidden_layers,
		size_t output_neurons,
		const std::vector<size_t>& neurons,
		std::function<VectorXd(const VectorXd&)> activation):
		activation(activation)
{
	std::srand(time(NULL));
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
