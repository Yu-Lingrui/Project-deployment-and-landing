##基础镜像
FROM 192.168.1.186:8099/evtrain/ubuntu18.04-cuda11.1-cudnn8-devel-train-test-pytorch1.10.0-openvino2021r3-base:v1.2

COPY . /usr/local/ev_sdk


RUN cd /usr/local/ev_sdk \
	&& cd /usr/local/ev_sdk && rm -rf build \
	&& mkdir -p build && cd build \
	&& cmake -DCMAKE_BUILD_TYPE=release .. && make -j4 && make install \
	&& cd /usr/local/ev_sdk/test && rm -rf build && mkdir -p build && cd build && cmake -DCMAKE_BUILD_TYPE=release .. && make -j4 && make install
	
ENV LD_LIBRARY_PATH=/usr/local/ev_sdk/lib/:$LD_LIBRARY_PATH




