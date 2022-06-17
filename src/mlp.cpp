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
	const std::vector<VectorXd>& S,
	const VectorXd& Sd, // Desired
	VectorXd& delta
	) const
{
	MatrixXd d(I, J);

	const auto& So = S.back();
	const auto& Shk = *(S.rbegin()+1);

	assert((size_t)So.size() == J);
	assert((size_t)Shk.size() == I);

	for(size_t j = 0; j < J; j++)
		delta(j) = So[j]-Sd[j];

	for(size_t i = 0; i < I; i++)
		for(size_t j = 0; j < J; j++)
			d(i,j) = delta(j)*Shk(i);

	return d;
}

// FIXME
MatrixXd MLP::derivada_hh(
	size_t I,
	size_t J,
	size_t k,
	const std::vector<VectorXd>& S,
	VectorXd& delta
	) const
{
	VectorXd tmp(J);
	MatrixXd d(I, J);
	const auto& Shk = S[k+1];
	const auto& Shkm1 = S[k];

	assert((size_t)Shk.size() == J);
	assert((size_t)Shkm1.size() == I);

	for(size_t j = 0; j < J; j++){
		double valTmp = 0.0;
		for(size_t z = 0; z < (size_t)delta.size(); z++){
			valTmp += delta(z)*W[k](j,z);
		}
		tmp(j) = valTmp*Shk(j)*(1.0-Shk(j));
	}

	for(size_t i = 0; i < I; i++){
		for(size_t j = 0; j < J; j++){
			d(i,j) = tmp(j)*Shkm1(i);
		}
	}

	delta = tmp;
	return d;
}

std::vector<VectorXd> MLP::full_forward(VectorXd C) const
{
	std::vector<VectorXd> Sh;

	Sh.push_back(C);
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

void MLP::train(const std::vector<VectorXd>& X, const std::vector<int>& y, size_t batch_size, double alpha)
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
	VectorXd delta(W.back().cols()); //cambiar

	for(ssize_t i = W.size()-1; i >= 0; i--)
	{
		size_t I = W[i].rows();
		size_t J = W[i].cols();

		if(i == (ssize_t)W.size() -1)
			W[i] -= alpha*derivada_ho(I, J, S, Sd, delta);
		else
			W[i] -= alpha*derivada_hh(I, J, i, S, delta);

		assert(delta.size() == W.back().cols());
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

double MLP::loss(const std::vector<VectorXd>& X, const std::vector<int>& true_y) const
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
