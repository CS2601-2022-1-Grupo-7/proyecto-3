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

#include <iostream>

#include "arguments.hpp"
#include "mlp.hpp"

int main(int argc, char** argv) {
	arguments args(argc, argv);

	std::cout
		<< "Path:" << args.dataset_path << '\n'
		<< "Characteristics:" << args.characteristics << '\n'
		<< "Hidden layers:" << args.hidden_layers << '\n'
		<< "Hidden layers:\n"
		;

	for(auto i: args.neurons)
		std::cout << '\t' << i << '\n';

	std::cout
		<< "Output neurons:" << args.output_neurons << '\n'
		<< "Activation function: "
		;

	switch (args.activation)
	{
		case arguments::type::RELU:
			std::cout << "RELU\n";
			break;

		case arguments::type::SIGMOID:
			std::cout << "Sigmoid\n";
			break;

		case arguments::type::TANH:
			std::cout << "Tanh\n";
			break;
	}

	MatrixXd m = MatrixXd::Random(3,3);
	std::cout << m << '\n';
	MLP mlp({m});
	VectorXd Shk(3);
	VectorXd So(3);
	VectorXd Sd(3);
	Shk << 1, 2, 3;
	So << 2, 2, 2;
	Sd << 2, 3, 4;

	std::cout << mlp.forward(Shk) << '\n';
	std::cout << m.rows() << '\n';
	mlp.backward(10, 0.5, mlp.forward(So), mlp.forward(Sd), mlp.forward(Shk));
}
