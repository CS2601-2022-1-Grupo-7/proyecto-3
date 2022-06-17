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
	size_t I,
	size_t J,
	std::span<VectorXd> S,
	const VectorXd& Sd, // Desired
	VectorXd& delta
	) const
{
	MatrixXd d(I, J);

	const auto& So = S.back();
	const auto& Shk = *(S.rbegin()+1);

	assert((size_t)So.size() == J);
	assert((size_t)Shk.size() == I);

	VectorXd S_part(So.size());

	for(size_t j = 0; j < J; j++)
		S_part(j) = ((So[j]-Sd[j])*So[j]*(1.0-So[j]));

	for(size_t i = 0; i < I; i++)
		for(size_t j = 0; j < J; j++)
			d(i,j) = S_part(j)*Shk(i);

	return d;
}

// FIXME
MatrixXd MLP::derivada_hh(
	const MatrixXd& w,
	size_t km1,
	VectorXd &delta,
	const VectorXd& Shk, // Hidden
	const VectorXd& Shkm1 // Hidden
	) const
{

	VectorXd tmp(w.cols());
	MatrixXd d(w.rows(), w.cols());

	for(size_t j = 0; j < (size_t)w.cols(); j++){
		double valTmp = 0.0;
		for(size_t k = 0; k < (size_t)delta.size(); k++){
			valTmp += delta(k)*W[km1+1](j,k);
		}
		tmp(j) = valTmp*Shk(j)*(1.0-Shk(j));
	}

	for(size_t i = 0; i < (size_t)w.rows(); i++){
		for(size_t j = 0; j < (size_t)w.cols(); j++){
			d(i,j) = tmp(j)*Shkm1(i);
		}
	}

	delta = tmp;
	return d;
}

std::vector<VectorXd> MLP::full_forward(VectorXd C) const
{
	std::vector<VectorXd> Sh;

	for(size_t i = 0; i < W.size(); i++)
	{
		C = C.transpose()*W[i];
		C = i == W.size()-1? softmax(C): activation(C);
		Sh.push_back(C);
	}

	return Sh;
}

VectorXd MLP::forward(VectorXd C) const
{
	for(size_t i = 0; i < W.size(); i++)
	{
		C = C.transpose()*W[i];
		C = i == W.size()-1? softmax(C): activation(C);
	}

	return C;
}

double calc_E(const VectorXd& Sd, const VectorXd& So)
{
	double E = 0.0;

	for(size_t i = 0; i < (size_t)So.size(); i++)
	{
		E += (Sd[i]*log(So[i]));
	}
	return -1.0*E;
}

std::tuple<VectorXd, double> MLP::testing(VectorXd C, int y, double b){
	VectorXd Sdd = class2vector(y);

	for(const auto& w: W)
	{
		C = activation(C.transpose()*w);
		for(int c=0; c<C.size(); c++){
			C[c] = C[c]+b;
		}
	}
	return {C, calc_E(Sdd, C)};
}

void MLP::train(std::span<VectorXd> X, std::span<int> y, size_t batch_size, double alpha)
{
	for(size_t b = 0; b < batch_size; b++)
	{
		size_t i = rand() % X.size();
		backward(alpha, X[i], y[i]);
	}
}

void MLP::backward(double alpha, const VectorXd& X, int y)
{
	auto S = full_forward(X);
	VectorXd Sd = class2vector(y);
	VectorXd delta(W[W.size()-1].cols()); //cambiar
	std::vector<MatrixXd> WT = W;

	for(ssize_t i = W.size()-1; i >= 0; i--)
	{
		size_t I = W[i].rows();
		size_t J = W[i].cols();

		if(i == (ssize_t)W.size() -1){
			WT[i] -= alpha*derivada_ho(I, J, S, Sd, delta);
		}
		else{
			WT[i] -= alpha*derivada_hh(W[i], i, delta, S[i+0], S[i]);
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
	if (hidden_layers != 0){
		MatrixXd m = MatrixXd::Random(features,neurons[0]);
		W.push_back(m);
		for (size_t i=0; i<hidden_layers-1; i++){
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

double MLP::loss(std::span<VectorXd> X, std::span<int> true_y) const
{
	assert(X.size() == true_y.size());

	double r = 0;

	for(size_t i = 0; i < X.size(); i++)
		r += calc_E(class2vector(true_y[i]), forward(X[i]));

	return r/X.size();
}

VectorXd MLP::class2vector(int _class) const
{
	size_t n = W.back().cols();
	VectorXd v(n);

	for(size_t i = 0; i < n; i++)
	{
		v[i] = (int)i == _class-1 ? 1 : 0;
	}

	return v;
}
