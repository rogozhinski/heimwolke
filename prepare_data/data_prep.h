#ifndef _DATA_PREP_H_
#define _DATA_PREP_H_

#include<string>
#include<map>
#include<vector>
#include<memory>


class DataPrep {

    private:
        std::string data_dir;
        float test_share;
        std::unique_ptr<std::map<std::string, std::string>> class_map = std::make_unique<std::map<std::string, std::string>> ();
        void collect_mapping();
        void write_pairs(std::vector<std::pair<std::string, std::string>> &pairs, std::string name);

    public:
        DataPrep (std::string data_dir_arg, float test_share_arg);
        void annotate (std::string anno_dir);
};


#endif
