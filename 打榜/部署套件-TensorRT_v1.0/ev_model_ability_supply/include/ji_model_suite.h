#ifndef JI_MODEL_SUITE_HPP
#define JI_MODEL_SUITE_HPP
#include<string>
#include "ev_common_data/ev_common_data.h"

class JiModelSuite
{
    private:

        JiModelSuite();

        JiModelSuite(const JiModelSuite&) = delete;

        JiModelSuite &operator= (const JiModelSuite&) = delete;

    public:
        
        static JiModelSuite& GetModelSuite();
            
        bool InitModelSuite(const std::string& config_file);
                    
        bool CreateModel(const std::string& uuid);
            
        bool RunInference(const std::string& uuid, ev_common_data_model_base_param_t* in, ev_common_data_model_base_param_t* out);
            
        bool DestroyModel(const std::string& uuid);

        bool GetVersion(std::string& version);      

        std::string GetUsedModels();  
};

#endif

