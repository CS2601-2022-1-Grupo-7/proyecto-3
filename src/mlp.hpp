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

	VectorXd softMax(VectorXd So);

	std::function<VectorXd(const VectorXd&)> activation;

	MatrixXd derivada_ho(
		const MatrixXd& w,
		size_t k,
		VectorXd& delta,
		const VectorXd& So, // Output
		const VectorXd& Sd, // Desired
		const VectorXd& Shk // Hidden
		);


	MatrixXd derivada_hh(
		const MatrixXd& w,
		size_t km1,
		VectorXd &delta,
		const VectorXd& Shk, // Hidden
		const VectorXd& Shkm1 // Hidden
		);

public:
	MLP(size_t features,
		size_t hidden_layers,
		size_t output_neurons,
		const std::vector<size_t>& neurons,
		std::function<VectorXd(const VectorXd&)> activation);

	VectorXd forward(VectorXd C);
	VectorXd semi_forward(VectorXd C);

	void backward(size_t epoch, double alpha, VectorXd So, VectorXd Sd, VectorXd Shk);

};
