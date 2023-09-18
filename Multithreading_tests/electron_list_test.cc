#include "electron_list.h"
#include <iostream>
#include <thread>
#include <omp.h>

void print_list_info(Electron_list& list)
{
    std::cout << "---List---" << std::endl;
    std::cout << "Num_electrones:" << list.electron_count << std::endl;
    std::cout << "Head:" << list.head <<std::endl;
    std::cout << "Tail:" << list.tail <<std::endl;
}

int main() {
    Electron_list global_list;
    
    std::cout << "Initial list_1:" << std::endl;
    print_list_nodes(global_list);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int tnr = omp_get_num_threads();

        Electron_list local_list;
        
        for (int i=0;i<3;i++)
        {
            std::shared_ptr<Electron> new_e_1 = std::make_shared<Electron>();
            new_e_1->face   = i*10+tid;
            new_e_1->phase  = i*10+tid;
            new_e_1->energy = i*10+tid;
            new_e_1->next   = nullptr;

            add_electron(local_list,new_e_1);
        }

        append_list_thread(global_list,local_list);
    }
    
    std::cout << std::endl << std::endl << "Final list_1:" << std::endl;
    print_list_nodes(global_list);

    return 0;
}
