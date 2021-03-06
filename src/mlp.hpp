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

#pragma once

#include <cmath>
#include <functional>
#include <iostream>
#include <vector>

#include <Eigen/Dense>

using Eigen::MatrixXd;
using Eigen::VectorXd;

class MLP
{
private:
	// Matrices http: https://eigen.tuxfamily.org/index.php?title=Main_Page
	std::vector<MatrixXd> W;
	//std::vector<VectorXd> Sh;

	std::function<VectorXd(const VectorXd&)> activation;

	MatrixXd derivada_ho(
		size_t I,
		size_t J,
		const std::vector<VectorXd>& S,
		const VectorXd& Sd, // Desired
		VectorXd& delta
		) const;


	MatrixXd derivada_hh(
		size_t I,
		size_t J,
		size_t k,
		const std::vector<VectorXd>& S,
		VectorXd& delta
		) const;

	VectorXd class2vector(int _class) const;
	std::vector<VectorXd> full_forward(VectorXd C) const;

public:
	MLP(size_t features,
		size_t hidden_layers,
		size_t output_neurons,
		const std::vector<size_t>& neurons,
		std::function<VectorXd(const VectorXd&)> activation);

	VectorXd forward(VectorXd C) const;

	void backward(double alpha, const VectorXd& X, int y);

	void train(const std::vector<VectorXd>& X, const std::vector<int>& y, size_t batch_size, double alpha);
	std::vector<int> predict(const std::vector<VectorXd>& X) const;
	double loss(const std::vector<VectorXd>& X, const std::vector<int>& true_y) const;
};
