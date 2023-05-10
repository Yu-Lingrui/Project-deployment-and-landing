cmake_minimum_required(VERSION 3.5)
project(ji)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_INSTALL_PREFIX "/usr/local/ev_sdk")
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
SET(CMAKE_BUILD_TYPE "Release")
#先编译三方库中的源码，并包含头文件路径
#jsoncpp相关头文件路径和源文件
include_directories(${CMAKE_SOURCE_DIR}/3rd/jsoncpp_simple/include)
add_subdirectory(${CMAKE_SOURCE_DIR}/3rd/jsoncpp_simple)
#配置点线框解析相关头文件路径和源文件
include_directories(${CMAKE_SOURCE_DIR}/3rd/wkt_parser/include)
add_subdirectory(${CMAKE_SOURCE_DIR}/3rd/wkt_parser)

# 使用pkg-config查找必要的依赖包
find_package(OpenCV REQUIRED)
find_library(glog REQUIRED)
set(ENV{PKG_CONFIG_PATH} "ENV{PKG_CONFIG_PATH}")
find_package(PkgConfig REQUIRED)
set(onnxruntime_INCLUDE_DIRS "/usr/local/ev_sdk/3rd/onnxruntime-linux-x64-1.6.0/include") #本demo所需的onnxruntime
include_directories(${onnxruntime_INCLUDE_DIRS})

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_RPATH}:/usr/local/ev_sdk/lib")

# 编译libji.so
aux_source_directory(${CMAKE_CURRENT_SOURCE_DIR}/src SRCS)
list(APPEND libji_SOURCES  ${SRCS})

add_library (${PROJECT_NAME} SHARED ${libji_SOURCES})
target_include_directories(${PROJECT_NAME}
        PUBLIC ${OpenCV_INCLUDE_DIR}
        PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
        )
target_link_libraries(${PROJECT_NAME}
        ${OpenCV_LIBS}
        glog
        WKTParser
        jsoncpp
        "/usr/local/ev_sdk/3rd/onnxruntime-linux-x64-1.6.0/lib/libonnxruntime.so"
        )

# 安装SampleDetector所依赖的onnxruntime库, 请在此处将自己依赖的库安装到/usr/local/ev_sdk/lib下。此处需注意，务必将相关依赖库都安装在该目录下，
# 避免运行时无法找到链接库（算法sdk被上层应用调用时依赖该/usr/local/ev_sdk/lib路径）
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/3rd/onnxruntime-linux-x64-1.6.0/lib/libonnxruntime.so DESTINATION lib)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/3rd/onnxruntime-linux-x64-1.6.0/lib/libonnxruntime.so.1.6.0 DESTINATION lib)

# Install libji.so
install(TARGETS ${PROJECT_NAME} DESTINATION lib)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/3rd/fonts DESTINATION lib)
# ---  END EV_SDK DEPENDENCIES  ---