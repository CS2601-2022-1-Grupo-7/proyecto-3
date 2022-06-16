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

#include <cmath>
#include <iostream>
#include <fstream>

#include "arguments.hpp"
#include "input.hpp"
#include "mlp.hpp"

std::function<VectorXd(const VectorXd&)> set_activation(arguments::type t)
{
	switch(t)
	{
		case arguments::type::RELU:
			return
				[](const VectorXd& v)
				{
					VectorXd Net = v;
					for(size_t i = 0; i < (size_t)Net.size(); i++)
					{
						Net(i) = std::max(Net(i), 0.0);
					}
					return Net;
				};

		case arguments::type::TANH:
			return
				[](const VectorXd& v)
				{
					VectorXd Net = v;
					for(size_t i = 0; i < (size_t)Net.size(); i++)
					{
						Net(i) = (1.0-exp(-2.0*Net(i)))/(1.0+exp(-2.0*Net(i)));
					}
					return Net;
				};

		case arguments::type::SIGMOID:
		default:
			return
				[](const VectorXd& v)
				{
					VectorXd Net = v;
					for(size_t i = 0; i < (size_t)Net.size(); i++)
					{
						Net(i) = 1.0/(1.0+exp(-Net(i)));
					}
					return Net;
				};
	}
}

int main(int argc, char** argv) {

	arguments args(argc, argv);
	input     i(args.json_file);

	MLP mlp(i.test_X.front().size(),
		args.hidden_layers,
		args.output_neurons,
		args.neurons,
		set_activation(args.activation)
	);

	std::vector<double> errorsTraing;
	std::vector<double> errorsTest;
	int epoch = 50;
	double totalProm = 0;
	double errorT = 0; 
	for (int k=0; k<epoch; k++){
		for (int j=0; j<args.batch_size; j++){
			std::srand(time(NULL));
			int randomTraining = rand() % i.test_X.size();
			errorT = mlp.training(0.01, i.train_X[randomTraining], i.train_y[randomTraining], 1);
			totalProm += errorT;
		}
		totalProm = totalProm/args.batch_size;
		errorsTraing.push_back(totalProm);
	}

	double accuracy = 0;
	double tmp = 0;
	for(size_t ii = 0; ii < i.test_X.size(); ii++)
	{
		auto [result, tmp] = mlp.testing(i.test_X[ii], i.test_y[ii], 0);
		int classR = std::max_element(result.begin(),result.end()) - result.begin() +1;
		errorsTest.push_back(tmp);
		std::cout<< "class: "<< i.test_y[ii]<< " result: " << classR<< std::endl;
		if (i.test_y[ii] == classR){
			accuracy +=1;
		}
	}
	std::ofstream myfileTest ("error.csv");
	myfileTest << "epoch,errorTraining" <<  '\n' ;
	for (int i=0; i<errorsTraing.size(); i++){
		myfileTest << i << "," << errorsTraing[i] <<  '\n' ;
	}
	myfileTest.close();

	// std::ofstream myfile ("error.txt");
	// for (int i=0; i<errorsTraing.size(); i++){
	// 	myfile << errorsTraing[i] <<  '\n' ;
	// }
	// myfile.close();
	std::cout << (accuracy/i.test_X.size()) * 100 << "%" << std::endl;
}
