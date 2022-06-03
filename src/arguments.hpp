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

#include <cstdio>
#include <vector>

class arguments
{
private:
	[[ noreturn ]]
	void usage(int exit_code) const;

	int argc;
	char** argv;

public:
	enum class type
	{
		SIGMOID,
		TANH,
		RELU
	};

	FILE*            json_file;
	int              hidden_layers;
	std::vector<int> neurons;
	int              output_neurons;
	type             activation;

	void parse();

	arguments(int argc, char** argv):
		argc(argc),
		argv(argv),
		json_file(nullptr)
	{
		parse();
	}

	~arguments()
	{
		fclose(json_file);
	}
};
