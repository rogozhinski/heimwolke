#include<iostream>
#include<memory>


#include"./data_prep.h"


int main() {
    std::unique_ptr<DataPrep> data_prep = std::make_unique<DataPrep>(
        "/home/user/heimwolke_root/CCSN_v2/", 
        0.2
    );
    data_prep->annotate("/home/user/heimwolke_root/data");
    return 0;
};
