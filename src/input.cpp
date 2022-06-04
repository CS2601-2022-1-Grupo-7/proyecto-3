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

#include <climits>
#include <iostream>
#include <stack>
#include <string>
#include <unordered_map>

#include <rapidjson/error/en.h>
#include <rapidjson/filereadstream.h>
#include <rapidjson/reader.h>

#include "input.hpp"

class json_parser:
	public rapidjson::BaseReaderHandler<rapidjson::UTF8<>, json_parser>
{
private:
	enum class state_t
	{
		before_json,
		before_inputs,
		before_array,
		before_matrix,
		before_vectors,
		in_array,
		in_vector
	};

	state_t state = state_t::before_json;

	char                      buffer[PIPE_BUF];
	rapidjson::Reader         reader;
	rapidjson::FileReadStream frs;
	input&                    i;

	struct new_state
	{
		state_t                        state;
		union
		{
			std::vector<VectorXd> input::* X_p;
			std::vector<int> input::*      y_p;
		};
	};

	using key_map_t = std::unordered_map<std::string, new_state>;

	static const key_map_t  key_map;
	std::vector<double>     v_buffer;

	std::vector<VectorXd> input::* X_p;
	std::vector<int> input::*      y_p;

public:
	json_parser(FILE* json_file, input& i):
		reader(),
		frs(json_file, buffer, sizeof(buffer)),
		i(i)
	{};

	void parse()
	{
		if(!reader.Parse(frs, *this))
		{
			rapidjson::ParseErrorCode e = reader.GetParseErrorCode();
			std::cerr << rapidjson::GetParseError_En(e) << '\n';
		}
	}

	bool StartObject()
	{
		switch(state)
		{
			case state_t::before_json:
				state = state_t::before_inputs;
				break;

			default:
				return false;
		}

		return true;
	}

	bool EndObject(rapidjson::SizeType)
	{
		switch(state)
		{
			case state_t::before_inputs:
				state = state_t::before_json;
				break;

			default:
				return false;
		}

		return true;
	}

	bool StartArray()
	{
		switch(state)
		{
			case state_t::before_matrix:
				state = state_t::before_vectors;
				break;

			case state_t::before_vectors:
				v_buffer.clear();
				state = state_t::in_vector;
				break;

			case state_t::before_array:
				state = state_t::in_array;
				break;

			default:
				return false;
		}

		return true;
	}

	bool EndArray(rapidjson::SizeType)
	{
		switch(state)
		{
			case state_t::in_vector:
				state = state_t::before_vectors;
				(i.*X_p).emplace_back(v_buffer.size());

				for(size_t ii = 0; ii < v_buffer.size(); ii++)
					(i.*X_p).back()[ii] = v_buffer[ii];

				break;

			case state_t::before_vectors:
				state = state_t::before_inputs;
				break;

			case state_t::in_array:
				state = state_t::before_inputs;
				break;

			default:
				return false;
		}

		return true;
	}

	bool Key(const char* str, rapidjson::SizeType size, bool)
	{
		switch(state)
		{
			case state_t::before_inputs:
			{
				auto it = key_map.find({str, size});

				if(it == key_map.end())
					return false;

				state = it->second.state;
				X_p = it->second.X_p;
				y_p = it->second.y_p;

				break;
			}

			default:
				return false;
		}

		return true;
	}

	bool String(const char* str, rapidjson::SizeType, bool)
	{
		switch(state)
		{
			case state_t::in_array:
				(i.*y_p).push_back(atoi(str));
				break;

			default:
				return false;
		}

		return true;
	}

	bool Double(double d)
	{
		switch(state)
		{
			case state_t::in_vector:
				v_buffer.push_back(d);
				break;

			default:
				return false;
		}

		return true;
	}

	bool Default()
	{
		return false;
	}
};

const json_parser::key_map_t json_parser::key_map =
{
	{"trainX",    { state_t::before_matrix, {.X_p = &input::train_X}}},
	{"testX",     { state_t::before_matrix, {.X_p = &input::test_X}}},
	{"validateX", { state_t::before_matrix, {.X_p = &input::validate_X}}},
	{"trainY",    { state_t::before_array,  {.y_p = &input::train_y}}},
	{"testY",     { state_t::before_array,  {.y_p = &input::test_y}}},
	{"validateY", { state_t::before_array,  {.y_p = &input::validate_y}}}
};

input::input(FILE* json_file)
{
	json_parser parser(json_file, *this);
	parser.parse();
}
