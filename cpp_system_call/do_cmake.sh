#!/bin/bash

BUILDDIR=./build

rm -rf ${BUILDDIR} && mkdir ${BUILDDIR}
pushd ${BUILDDIR}
cmake .. || exit
make -j || exit
popd
