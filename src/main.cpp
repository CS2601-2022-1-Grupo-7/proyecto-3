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
#include "utils.hpp"

int main(int argc, char** argv) {
	std::srand(time(NULL));

	arguments args(argc, argv);
	input     i(args.json_file);

	MLP mlp(i.test_X.front().size(),
		args.hidden_layers,
		args.output_neurons,
		args.neurons,
		set_activation(args.activation)
	);

	std::ofstream csv_file ("error.csv");
	std::ofstream test_csv_file ("testing.csv");

	if(!csv_file.is_open() || !test_csv_file.is_open())
		return EXIT_FAILURE;

	csv_file << "epoch,train,validation\n";

	for(size_t e = 0; e < args.epochs; e++)
	{
		mlp.train(i.train_X, i.train_y, args.batch_size, args.alpha);

		csv_file
			<< e+1
			<< ','
			<< mlp.loss(i.train_X, i.train_y)
			<< ','
			<< mlp.loss(i.validate_X, i.validate_y)
			<< '\n'
		;
	}

	auto yp = mlp.predict(i.test_X);
	test_csv_file << "y,yp\n";

	for(size_t ii = 0; ii < yp.size(); ii++)
		test_csv_file << i.test_y[ii] << ',' << yp[ii] << '\n';

	return EXIT_SUCCESS;
}
