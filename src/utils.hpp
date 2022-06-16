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

#include "arguments.hpp"

#include <functional>

#include <Eigen/Dense>

using Eigen::VectorXd;

VectorXd relu(const VectorXd& v);
VectorXd _tanh(const VectorXd& v);
VectorXd sigmoid(const VectorXd& v);
VectorXd softmax(const VectorXd& v);

std::function<VectorXd(const VectorXd&)> set_activation(arguments::type t);
VectorXd class2vector(int _class, size_t n);
