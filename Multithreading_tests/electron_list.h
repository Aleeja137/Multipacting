#ifndef ELECTRON_LIST_H
#define ELECTRON_LIST_H

#include <mutex>
#include <memory>
#include <iostream>

struct Electron {
    int face = 0;
    double energy = 0.0;
    double phase = 0.0;
    // double power;
    // int generation;
    std::shared_ptr<Electron> next = nullptr;
};

struct Electron_list {
    std::shared_ptr<Electron> head = nullptr;
    std::shared_ptr<Electron> tail = nullptr;
    unsigned int electron_count = 0;
    std::mutex mtx;
};

std::shared_ptr<Electron> get_electron(Electron_list& list);
std::shared_ptr<Electron> get_electron_thread(Electron_list& list);
void add_electron(Electron_list& list, std::shared_ptr<Electron> electron);
void append_list(Electron_list& original_list, Electron_list& new_list);
void append_list_thread(Electron_list& original_list, Electron_list& new_list);
void resetear_lista(Electron_list& list);
void copiar_lista(Electron_list& lista_destino, Electron_list& lista_origen);

// Testing
void print_list_nodes(Electron_list& list);
void print_electron(std::shared_ptr<Electron> e);

#endif // ELECTRON_LIST_H