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

#include "utils.hpp"

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

VectorXd class2vector(int _class, size_t n)
{
	VectorXd v(n);

	for(size_t i = 0; i < n; i++)
	{
		v[i] = (int)i == _class-1 ? 1 : 0;
	}

	return v;
}
