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
#include <getopt.h>
#include <iostream>

#include "arguments.hpp"

void arguments::usage(int exit_code) const
{
	std::cout
		<< "Usage: " << argv[0] << " PATH\n"
		<< "\t-h, --help Show this help\n"
		;

	exit(exit_code);
}

void arguments::parse()
{
	if(argc < 2)
		usage(EXIT_FAILURE);

	int c;

	static const char shortopts[] = "h";
	static const option options[] =
	{
		{"help",   no_argument,       nullptr, 'h'},
		{nullptr,  0,                 nullptr, 0},
	};

	while((c = getopt_long(argc, argv, shortopts, options, nullptr)) != -1)
	{
		switch(c)
		{
			case 'h':
				usage(EXIT_SUCCESS);

			case '?':
				exit(EXIT_FAILURE);

			default:
				usage(EXIT_FAILURE);
		}
	}

	for(int i = optind; i < argc; i++)
		dataset_path = argv[i];
}
