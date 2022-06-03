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

#include <cstdlib>
#include <cstring>
#include <getopt.h>
#include <iostream>
#include <unistd.h>

#include "arguments.hpp"

void arguments::usage(int exit_code) const
{
	std::cout
		<< "Usage: " << argv[0] << " JSON\n"
		<< "\t-h, --help                 Show this help.\n"
		<< "\t-l, --hidden-layers=N      Number of hidden layers.\n"
		<< "\t-n, --neurons=N,...        Number of neurons in each layer.\n"
		<< "\t-N, --output-neurons=N     Number of neurons in the output layer.\n"
		<< "\t-s, --sigmoid              Select Sigmoid activation function.\n"
		<< "\t-t, --tanh                 Select Tanh activation function.\n"
		<< "\t-r, --relu                 Select RELU activation function.\n"
		;

	exit(exit_code);
}

void arguments::parse()
{
	if(argc < 2)
		usage(EXIT_FAILURE);

	int c;

	static const char shortopts[] = "hl:n:N:str";
	static const option options[] =
	{
		{"help",            no_argument,       nullptr, 'h'},
		{"hidden-layers",   required_argument, nullptr, 'l'},
		{"neurons",         required_argument, nullptr, 'n'},
		{"output-neurons",  required_argument, nullptr, 'N'},
		{"sigmoid",         no_argument,       nullptr, 's'},
		{"tanh",            no_argument,       nullptr, 't'},
		{"relu",            no_argument,       nullptr, 'r'},
		{nullptr,           0,                 nullptr, 0},
	};

	while((c = getopt_long(argc, argv, shortopts, options, nullptr)) != -1)
	{
		switch(c)
		{
			case 'h':
				usage(EXIT_SUCCESS);

			case 'l':
				hidden_layers = atoi(optarg);
				break;

			case 'n':
				static const char delim[] = " ,\t\n";
				char* buffer;

				for(char* token = strtok_r(optarg, delim, &buffer);
					token;
					token = strtok_r(nullptr, delim, &buffer)
				)
				{
					neurons.push_back(atoi(token));
				}
				break;

			case 'N':
				output_neurons = atoi(optarg);
				break;

			case 's':
				activation = type::SIGMOID;
				break;

			case 't':
				activation = type::TANH;
				break;

			case 'r':
				activation = type::RELU;
				break;

			case '?':
				exit(EXIT_FAILURE);

			default:
				usage(EXIT_FAILURE);
		}
	}

	const char* json_path = nullptr;

	for(int i = optind; i < argc; i++)
		json_path = argv[i];

	json_file = json_path ? fopen(json_path, "r"): fdopen(dup(STDIN_FILENO), "r");

	if(!json_file)
	{
		perror(json_path);
		exit(EXIT_FAILURE);
	}
}
