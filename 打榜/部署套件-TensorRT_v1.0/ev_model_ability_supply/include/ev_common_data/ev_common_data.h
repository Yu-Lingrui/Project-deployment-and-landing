#ifndef EV_COMMON_DATA_H
#define EV_COMMON_DATA_H

#include <sys/stat.h> 
#include "glog/logging.h"

#include "ev_common_data/ev_common_data.h"

#define EMSLOG(t) LOG(t)<<"[EMSLOG] "
#define EMSLOG_IF(t,f) LOG_IF(t,f)<<"[EMSLOG] "
#define EMSLOG_FIRST_N(t, i) LOG_FIRST_N(t, i)<<"[EMSLOG] "
#define EMSLOG_EVERY_N(t, i) LOG_EVERY_N(t, i)<<"[EMSLOG] "
#define EMSLOG_IF_EVERY_N(t, f, i) LOG_IF_EVERY_N(t, f, i)<<"[EMSLOG] "


#ifdef DLOG
  #define DLOGP(t) LOG(t) << "[DLOG] "
#else
  #define DLOGP(t) 0 && std::cout
#endif



typedef enum ev_common_data_status_t {
  EV_COMMON_DATA_SUCCESS          = 0,
  EV_COMMON_DATA_E_INVALID_ARG    = 1,
  EV_COMMON_DATA_E_NOT_SUPPORTED  = 2,
  EV_COMMON_DATA_E_OUT_OF_RANGE   = 3,
  EV_COMMON_DATA_E_OUT_OF_MEMORY  = 4,
  EV_COMMON_DATA_E_FILE_NOT_EXIST = 5,
  EV_COMMON_DATA_E_FAIL           = 6,
  EV_COMMON_DATA_E_REPEATED_OP    = 17
} ecd_status_t;

// 像素格式
typedef enum ev_common_data_pixel_t{
  EV_COMMON_DATA_PIXEL_BGR      = 0,
  EV_COMMON_DATA_PIXEL_RGB      = 1,
  EV_COMMON_DATA_PIXEL_GRAY     = 2,
  EV_COMMON_DATA_PIXEL_NV12     = 3,
  EV_COMMON_DATA_PIXEL_NV21     = 4,
  EV_COMMON_DATA_PIXEL_BGRA     = 5
} emd_pixel_t;

// 数据格式
typedef enum ev_common_data_data_t{
  EV_COMMON_DATA_DATA_FLOAT     = 0,
  EV_COMMON_DATA_DATA_HALF      = 1,
  EV_COMMON_DATA_DATA_UINT8     = 2,
  EV_COMMON_DATA_DATA_INT32     = 3
} emd_data_t;

// 矩形感兴趣区域
typedef struct ev_common_data_rect_t {
  float left;
  float top;
  float right;
  float bottom;
} emd_rect_t;

// 图像结构
typedef struct ev_common_data_mat_t {
  void* data;
  int height;
  int width;
  int channel;
  int aligned_width;
  int aligned_height;
  emd_pixel_t format;
  emd_data_t type;
  emd_rect_t range;  
} emd_mat_t;

// 推理的输入输出参数
typedef struct ev_common_data_model_base_param_t {
  emd_mat_t *mat;
  int mat_num;
  char* desc;  
} emd_model_base_param_t;


// 推理的输入输出参数
typedef enum ev_common_data_file_type {
  EV_COMMON_DATA_FILE_TYPE_FILE         = 0,
  EV_COMMON_DATA_FILE_TYPE_FOLDER       = 1,
  EV_COMMON_DATA_FILE_TYPE_NOT_EXIST    = 2  
} ecd_file_type;



static ecd_status_t ev_check_emd_mat_t(emd_mat_t* t)
{
  if(t->data == NULL)
  {
    return EV_COMMON_DATA_E_INVALID_ARG;
  }

  if(t->range.top < 0 || t->range.top >= t->height ||
     t->range.bottom < 0 || t->range.bottom >= t->height ||
     t->range.left < 0 || t->range.left >= t->width ||
     t->range.right < 0 || t->range.right >= t->width ) 
  {
    return EV_COMMON_DATA_E_OUT_OF_RANGE;
  }
  
  return EV_COMMON_DATA_SUCCESS;
}


template<typename T>
T* ev_create_model_data()
{
  return malloc(sizeof(T));
}
 
template<typename T>
ecd_status_t ev_destroy_model_data(T* t)
{
  if(t == NULL)
  {
    return EV_COMMON_DATA_E_INVALID_ARG;
  }
  free(t);
  t = NULL;
  return EV_COMMON_DATA_SUCCESS;
}

//只判断普通文件和文件夹，不考虑链接的情况
static ecd_file_type check_file_type(const char *file_name)
{
    ecd_file_type ft = EV_COMMON_DATA_FILE_TYPE_NOT_EXIST;
    struct stat file_stat;
    int result = stat(file_name, &file_stat);
    if( (result == 0)  )
    {
      if( (file_stat.st_mode & S_IFMT) == S_IFDIR ) {
        ft = EV_COMMON_DATA_FILE_TYPE_FOLDER;
      } else if( (file_stat.st_mode & S_IFMT) == S_IFREG ){
        ft = EV_COMMON_DATA_FILE_TYPE_FILE;
      }
    }
    return ft;
}




#endif